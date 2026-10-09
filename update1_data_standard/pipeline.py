"""Update 1 data pipeline. No LID model calls; raw files are never changed."""
import argparse
import ast
import csv
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

VERSION = '1.0'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def new_dir(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise ValueError(f'Output directory must be empty: {p}')
    p.mkdir(parents=True, exist_ok=True)
    return p


def unique(rows, key):
    result = {}
    for r in rows:
        if r[key] in result:
            raise ValueError(f'Duplicate {key}: {r[key]}')
        result[r[key]] = r
    return result


def flatten(value, path=()):
    """Accept strings, lists, and documented instances containers only."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from flatten(child, path + (i,))
    elif isinstance(value, dict) and 'instances' in value:
        yield from flatten(value['instances'], path + ('instances',))
    else:
        raise ValueError(f'Unsupported text container at {path}')


def extract(args):
    source = Path(args.input)
    lang_a, lang_b = args.languages.split(',')
    if lang_a == lang_b or '_' in lang_a + lang_b:
        raise ValueError('Use two distinct agreed language codes')
    pair = f'{lang_a}_{lang_b}'
    checksum = digest(source.read_bytes())
    texts, issues = [], []
    with source.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if 'data_generation_result' not in (reader.fieldnames or []):
            raise ValueError('Missing data_generation_result')
        count = 0
        for row_number, row in enumerate(reader, 1):
            if args.limit and row_number > args.limit:
                break
            count += 1
            rid = f'{pair}:{checksum[:12]}:{row_number:06d}'
            try:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError('Malformed CSV columns')
                cell = row['data_generation_result']
                if len(cell) > 1_000_000:
                    raise ValueError('Oversized cell requires review')
                try:
                    obj = json.loads(cell)
                except json.JSONDecodeError:
                    obj = ast.literal_eval(cell)  # Literal data only, never eval().
                items = list(flatten(obj))
                if not items:
                    raise ValueError('Empty text list')
                for item_number, (source_path, text) in enumerate(items, 1):
                    sid = f'{rid}:{item_number:03d}'
                    if not text.strip():
                        issues.append({'id': sid, 'severity': 'error', 'reason': 'empty_text'})
                        continue
                    texts.append({'schema_version': VERSION, 'sequence_id': sid,
                                  'record_id': rid, 'item_id': item_number,
                                  'language_pair': pair, 'languages': [lang_a, lang_b],
                                  'source_file': source.name, 'source_row': row_number,
                                  'source_path': list(source_path), 'text': text,
                                  'text_sha256': digest(text.encode('utf-8'))})
                    if '\\' in text or text.startswith(('"', "'")):
                        issues.append({'id': sid, 'severity': 'warning',
                                       'reason': 'possible_wrapping_or_escape_review_no_auto_change'})
            except (ValueError, SyntaxError, TypeError, RecursionError) as e:
                issues.append({'id': rid, 'severity': 'error', 'reason': str(e)})
    out = new_dir(args.out)
    write_jsonl(out / 'texts.jsonl', texts)
    write_jsonl(out / 'lid.template.jsonl', [dict(sequence_id=r['sequence_id'],
               text_sha256=r['text_sha256'], annotation_source='UNSET',
               review_status='unreviewed', reviewer='',
               segments=[{'text': r['text'], 'lang': 'UNKNOWN'}]) for r in texts])
    write_jsonl(out / 'extract_issues.jsonl', issues)
    write_json(out / 'extract_manifest.json', {'schema_version': VERSION,
               'source_file': source.name, 'source_sha256': checksum,
               'language_pair': pair, 'input_records': count, 'output_items': len(texts),
               'limit': args.limit, 'issues': len(issues),
               'sequence_policy': 'one_extracted_item_no_concatenation'})
    print(f'Extracted {len(texts)} items; {len(issues)} issues. Output: {out}')


def validate_lid(textrow, lid):
    if lid['text_sha256'] != textrow['text_sha256']:
        raise ValueError('Text hash mismatch')
    if lid.get('annotation_source') in (None, '', 'UNSET'):
        raise ValueError('Set annotation_source to the actual method/version')
    if lid.get('review_status') not in ('unreviewed', 'reviewed', 'needs_review'):
        raise ValueError('Invalid review_status')
    if lid['review_status'] == 'reviewed' and not lid.get('reviewer'):
        raise ValueError('Reviewed data requires reviewer name')
    allowed = set(textrow['languages']) | {'NEUTRAL', 'UNKNOWN'}
    segments = lid['segments']
    if not segments or ''.join(s['text'] for s in segments) != textrow['text']:
        raise ValueError('Segments must reproduce original text exactly, including spaces')
    labels, spans, start = [], [], 0
    for segment in segments:
        part, lang = segment['text'], segment['lang']
        if not part or lang not in allowed:
            raise ValueError('Empty segment or invalid language')
        labels.extend([lang] * len(part))
        spans.append({'start': start, 'end': start + len(part), 'lang': lang})
        start += len(part)
    return labels, spans


def align(text, char_labels, offsets, token_ids, unk_id, neutral_policy):
    """Only preceding token state is used to resolve NEUTRAL; never look ahead."""
    langs, reasons, previous = [], [], 'UNKNOWN'
    for token_id, (a, b) in zip(token_ids, offsets):
        found = {char_labels[i] for i in range(a, b) if not text[i].isspace()}
        reason = ''
        if token_id == unk_id:
            lang, reason = 'UNKNOWN', 'unknown_token'
        elif b <= a or not found:
            lang, reason = 'UNKNOWN', 'empty_offset'
        elif 'UNKNOWN' in found:
            lang, reason = 'UNKNOWN', 'unknown_lid'
        else:
            language_set = found - {'NEUTRAL'}
            if len(language_set) > 1:
                lang, reason = 'UNKNOWN', 'cross_language_subword'
            elif len(language_set) == 1:
                lang = next(iter(language_set))
            elif neutral_policy == 'inherit_previous' and previous != 'UNKNOWN':
                lang, reason = previous, 'neutral_inherited_previous'
            else:
                lang, reason = 'UNKNOWN', 'neutral_unresolved'
        langs.append(lang)
        reasons.append(reason)
        previous = lang
    return langs, reasons


def labels(langs, end_policy):
    n = len(langs)
    switch, duration, lengths, reasons = [-1]*n, [-1]*n, [None]*n, ['']*n
    ends = [n]*n
    for i in range(n-2, -1, -1):
        ends[i] = ends[i+1] if langs[i] == langs[i+1] else i+1
    for t in range(n):
        if t == n-1:
            reasons[t] = 'no_next_token'
        elif 'UNKNOWN' in (langs[t], langs[t+1]):
            reasons[t] = 'ambiguous_switch_boundary'
        else:
            switch[t] = int(langs[t] != langs[t+1])
            if not switch[t]:
                reasons[t] = 'no_switch'
                continue
            stop = ends[t+1]
            if stop < n and langs[stop] == 'UNKNOWN':
                reasons[t] = 'duration_unknown_boundary'
            elif stop == n and end_policy == 'censored':
                reasons[t] = 'duration_censored_at_item_end'
            else:
                length = stop-(t+1)
                lengths[t] = length
                duration[t] = 0 if length <= 2 else 1 if length <= 6 else 2
    return switch, duration, lengths, reasons


def tree_hash(path):
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob('*')):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def prepare_tokenizer(args):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    commit = HfApi().model_info(args.model, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=commit,
                                               use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError('Fast tokenizer required for offsets')
    out = new_dir(args.out)
    tokenizer.save_pretrained(out / 'tokenizer')
    write_json(out / 'tokenizer_manifest.json', {'model': args.model, 'revision': commit,
               'tokenizer_sha256': tree_hash(out / 'tokenizer')})
    print('Frozen tokenizer saved:', out)


def validate_row(r):
    n = len(r['input_ids'])
    keys = ['tokens', 'offsets', 'language_ids', 'alignment_notes', 'switch_labels',
            'switch_valid_mask', 'duration_labels', 'duration_valid_mask',
            'duration_lengths', 'label_notes']
    if any(len(r[k]) != n for k in keys):
        raise ValueError('Per-token arrays have unequal lengths')
    expected = labels(r['language_ids'], r['end_policy'])
    for k, value in zip(['switch_labels', 'duration_labels', 'duration_lengths', 'label_notes'], expected):
        if r[k] != value:
            raise ValueError(f'Incorrect {k}')
    if r['switch_valid_mask'] != [v >= 0 for v in r['switch_labels']]:
        raise ValueError('Incorrect switch mask')
    if r['duration_valid_mask'] != [v >= 0 for v in r['duration_labels']]:
        raise ValueError('Incorrect duration mask')
    if digest(r['text'].encode()) != r['text_sha256']:
        raise ValueError('Text checksum mismatch')
    for a, b in r['offsets']:
        if not 0 <= a <= b <= len(r['text']):
            raise ValueError('Offset outside original text')


def summary(rows):
    valid = sum(sum(r['switch_valid_mask']) for r in rows)
    switches = sum(r['switch_labels'].count(1) for r in rows)
    counts = Counter(v for r in rows for v in r['duration_labels'] if v >= 0)
    total = sum(counts.values())
    return {'sequences': len(rows), 'valid_positions': valid, 'switches': switches,
            'switch_rate': switches/valid if valid else None,
            'duration_class_counts': {str(i): counts[i] for i in range(3)},
            'duration_class_proportions': {str(i): counts[i]/total if total else None for i in range(3)},
            'switches_without_duration': switches-total,
            'review_status_counts': dict(Counter(r['review_status'] for r in rows)),
            'unknown_subwords': sum(r['language_ids'].count('UNKNOWN') for r in rows)}


def tokenize(args):
    import importlib.metadata
    from transformers import AutoTokenizer
    texts = unique(read_jsonl(args.texts), 'sequence_id')
    lids = unique(read_jsonl(args.lid), 'sequence_id')
    if set(texts) != set(lids):
        raise ValueError('LID IDs must exactly match texts IDs; create an explicit pilot texts file for subsets')
    asset = Path(args.tokenizer_dir)
    tm = json.loads((asset / 'tokenizer_manifest.json').read_text())
    if tree_hash(asset / 'tokenizer') != tm['tokenizer_sha256']:
        raise ValueError('Frozen tokenizer files changed')
    tok = AutoTokenizer.from_pretrained(asset / 'tokenizer', use_fast=True,
                                        local_files_only=True, trust_remote_code=False)
    if not tok.is_fast:
        raise ValueError('Fast tokenizer required')
    records, issues = [], []
    for sid, textrow in texts.items():
        lid = lids[sid]
        chars, spans = validate_lid(textrow, lid)
        if lid['review_status'] != 'reviewed' and not args.allow_unreviewed:
            raise ValueError(f'{sid}: not reviewed; explicit --allow-unreviewed required for pseudo labels')
        enc = tok(textrow['text'], add_special_tokens=False, truncation=False,
                  padding=False, return_offsets_mapping=True, return_attention_mask=False,
                  return_token_type_ids=False)
        if not enc['input_ids']:
            raise ValueError(f'{sid}: empty tokenization')
        langs, notes = align(textrow['text'], chars, enc['offset_mapping'],
                             enc['input_ids'], tok.unk_token_id, args.neutral_policy)
        sw, dur, lengths, reasons = labels(langs, args.end_policy)
        r = dict(textrow, tokens=tok.convert_ids_to_tokens(enc['input_ids']),
                 input_ids=enc['input_ids'], offsets=enc['offset_mapping'],
                 language_ids=langs, language_spans=spans, alignment_notes=notes,
                 switch_labels=sw, switch_valid_mask=[v >= 0 for v in sw],
                 duration_labels=dur, duration_valid_mask=[v >= 0 for v in dur],
                 duration_lengths=lengths, label_notes=reasons,
                 tokenizer_sha256=tm['tokenizer_sha256'], end_policy=args.end_policy,
                 neutral_policy=args.neutral_policy, annotation_source=lid['annotation_source'],
                 review_status=lid['review_status'], reviewer=lid.get('reviewer', ''))
        validate_row(r)
        records.append(r)
        for i, reason in enumerate(notes):
            if langs[i] == 'UNKNOWN':
                issues.append({'sequence_id': sid, 'position': i, 'reason': reason})
    out = new_dir(args.out)
    write_jsonl(out / 'labeled_tokens.jsonl', records)
    write_jsonl(out / 'token_issues.jsonl', issues)
    write_json(out / 'stats.json', summary(records))
    write_json(out / 'manifest.json', {'schema_version': VERSION, 'tokenizer': tm,
               'neutral_policy': args.neutral_policy, 'end_policy': args.end_policy,
               'texts_sha256': digest(Path(args.texts).read_bytes()),
               'lid_sha256': digest(Path(args.lid).read_bytes()),
               'pipeline_sha256': digest(Path(__file__).read_bytes()),
               'allow_unreviewed': args.allow_unreviewed,
               'versions': {p: importlib.metadata.version(p) for p in ['transformers', 'tokenizers']}})
    print('Labeled sequences:', len(records), 'Output:', out)


def split(args):
    rows = [r for path in args.inputs for r in read_jsonl(path)]
    unique(rows, 'sequence_id')
    if not rows:
        raise ValueError('No sequences')
    signatures = {(r['tokenizer_sha256'], r['neutral_policy'], r['end_policy']) for r in rows}
    if len(signatures) != 1:
        raise ValueError('Mixed tokenizers or label policies')
    for r in rows:
        validate_row(r)
    # Union original records that share normalized exact text, even across language pairs.
    parent = {r['record_id']: r['record_id'] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seen = {}
    for r in rows:
        key = ' '.join(unicodedata.normalize('NFC', r['text']).split())
        rid = r['record_id']
        if key in seen:
            a, b = find(rid), find(seen[key])
            parent[max(a, b)] = min(a, b)
        else:
            seen[key] = rid
    groups = defaultdict(list)
    for r in rows:
        groups[find(r['record_id'])].append(r)
    strata = defaultdict(list)
    for gid, items in groups.items():
        strata[tuple(sorted({r['language_pair'] for r in items}))].append(gid)
    assignment = {}
    for key, ids in sorted(strata.items()):
        ids.sort()
        random.Random(f'{args.seed}:{key}').shuffle(ids)
        n = len(ids)
        if n < 3:
            assignment.update({gid: 'train' for gid in ids})
            continue
        nv = nt = max(1, round(n*.1))
        for i, gid in enumerate(ids):
            assignment[gid] = 'validation' if i < nv else 'test' if i < nv+nt else 'train'
    forced = set()
    if getattr(args, 'train_records', None):
        forced = {s.strip() for s in Path(args.train_records).read_text(encoding='utf-8').splitlines() if s.strip()}
        if forced - set(parent):
            raise ValueError('Unknown record_id in train-records file')
        for rid in forced:
            assignment[find(rid)] = 'train'
    out = new_dir(args.out)
    split_rows = [{'sequence_id': r['sequence_id'], 'record_id': r['record_id'],
                   'group_id': find(r['record_id']), 'split': assignment[find(r['record_id'])]}
                  for r in rows]
    write_jsonl(out / 'splits.jsonl', split_rows)
    write_jsonl(out / 'labeled_tokens.jsonl', rows)
    write_json(out / 'split_manifest.json', {'seed': args.seed, 'target_ratios': [0.8, 0.1, 0.1],
               'counts': dict(Counter(r['split'] for r in split_rows)),
               'by_pair': {pair: dict(Counter(assignment[find(r['record_id'])] for r in rows
                           if r['language_pair'] == pair)) for pair in sorted({r['language_pair'] for r in rows})},
               'inputs': {str(p): digest(Path(p).read_bytes()) for p in args.inputs},
               'forced_train_records': sorted(forced),
               'duplicate_policy': 'NFC+whitespace exact duplicates union original records',
               'limitation': 'Near duplicates are NOT automatically detected. Strata with fewer than 3 groups go to train.'})
    print('Splits:', dict(Counter(r['split'] for r in split_rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--input', required=True)
    p.add_argument('--languages', required=True, help='Example: fr,en; agree codes with team')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', required=True)
    p.set_defaults(func=extract)
    p = sub.add_parser('prepare-tokenizer')
    p.add_argument('--model', required=True)
    p.add_argument('--revision', default='main')
    p.add_argument('--out', required=True)
    p.set_defaults(func=prepare_tokenizer)
    p = sub.add_parser('tokenize')
    for name in ['texts', 'lid', 'tokenizer-dir', 'out']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--neutral-policy', choices=['inherit_previous', 'mask'], required=True)
    p.add_argument('--end-policy', choices=['observed', 'censored'], required=True)
    p.add_argument('--allow-unreviewed', action='store_true')
    p.set_defaults(func=tokenize)
    p = sub.add_parser('split')
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--seed', type=int, default=6140)
    p.add_argument('--train-records', help='UTF-8 file, one original record_id per line; forces whole duplicate group to train')
    p.add_argument('--out', required=True)
    p.set_defaults(func=split)
    p = sub.add_parser('validate')
    p.add_argument('--input', required=True)
    def validate(args):
        rows = read_jsonl(args.input)
        unique(rows, 'sequence_id')
        for row in rows:
            validate_row(row)
        print(f'PASS: {len(rows)} sequences. This checks structure/rules, NOT linguistic accuracy.')
    p.set_defaults(func=validate)
    args = parser.parse_args()
    if getattr(args, 'limit', 0) < 0:
        parser.error('--limit must be nonnegative')
    args.func(args)


if __name__ == '__main__':
    main()
