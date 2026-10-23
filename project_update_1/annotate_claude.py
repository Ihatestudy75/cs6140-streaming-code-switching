"""Word-level LID labeled by Claude, for es_en, vi_en and the Latin-letter words of hi_en.

The model never writes or copies the text. The steps, run from the repository root:
  prepare   split each item into words and write batch files for the labeler
  (label)   Claude writes batch_NNN.labels.jsonl for every batch_NNN.input.jsonl, following
            claude_lid_instructions.md (a word-level version of LID_PROMPT.md v1.0)
  collect   list the words that the name and loanword rules govern, for one consistent pass
  (review)  one labeler writes a decision n/e/u for every candidate
  apply     copy the work dir with the decisions applied; record every change
  assemble  map the per-word codes back onto the original text and write lid.jsonl

The labeling and review steps were done in Claude Code (interactive session, subagents), so they
are not part of run_unicode_claude.py; it rebuilds the delivered labels from their character offsets.

A word is a run of letters (combining marks included) that may contain internal apostrophes or
hyphens. Only words get a language. Everything else (whitespace, punctuation, symbols, digits) is
NEUTRAL, the same character rule as annotate_lingua.py. Codes: n = native, e = en, u = UNKNOWN.

--script-native (hi_en: Hindi written in Devanagari or in Latin letters): characters start from the
annotate_script.py labels, and only all-Latin words are sent to the labeler.

collect candidates: words coded u; capitalized words that do not start the item (except "I");
English insertions inside native context (an e word whose neighbours are both n); and every
occurrence of a lowercase word that received both e and n somewhere, except short function words
that are true homographs in the two languages.
"""
import argparse
import datetime
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotate_script  # noqa: E402
from pipeline import read_jsonl, write_jsonl, write_json, new_dir, validate_lid, digest  # noqa: E402

LETTER = r'(?:[^\W\d_]|[̀-ͯ])'
WORD = re.compile(rf"{LETTER}+(?:['’‐‑-]{LETTER}+)*")
PROMPT_VERSION = 'v1.0'
HOMOGRAPHS = {'a', 'en', 'me', 'no', 'he', 'era', 'has', 'son', 'sin', 'come', 'tan', 'red', 'fin', 'pan',
              'mar', 'dice', 'once', 'e', 'o', 'y', 'i'}


def is_latin_word(word):
    return all('LATIN' in unicodedata.name(ch, '') for ch in word if ch.isalpha())


def words(text, script_native=False):
    spans = [(m.start(), m.end()) for m in WORD.finditer(text)]
    return [(a, z) for a, z in spans if is_latin_word(text[a:z])] if script_native else spans


def native_of(row):
    return next(c for c in row['languages'] if c != 'en')


def prepare(args):
    rows = read_jsonl(args.texts)
    out = new_dir(args.out)
    index = []
    for b in range(0, len(rows), args.batch_size):
        name = f'batch_{b // args.batch_size + 1:03d}'
        items = []
        for i, row in enumerate(rows[b:b + args.batch_size]):
            item_id = b + i + 1
            index.append({'id': item_id, 'sequence_id': row['sequence_id'], 'batch': name})
            items.append({'id': item_id, 'text': row['text'],
                          'words': [row['text'][a:z] for a, z in words(row['text'], args.script_native)]})
        write_jsonl(out / f'{name}.input.jsonl', items)
    write_jsonl(out / 'index.jsonl', index)
    write_json(out / 'prepare_manifest.json', {'texts': args.texts, 'texts_sha256': digest(Path(args.texts).read_bytes()),
               'items': len(rows), 'batch_size': args.batch_size, 'script_native': args.script_native,
               'batches': (len(rows) + args.batch_size - 1) // args.batch_size,
               'script_sha256': digest(Path(__file__).read_bytes())})
    print(f'{len(rows)} items -> {out}')


def load(work):
    items = {}
    for p in sorted(work.glob('batch_*.input.jsonl')):
        for r in read_jsonl(p):
            items[r['id']] = dict(r, batch=p.name.split('.')[0])
        for r in read_jsonl(work / p.name.replace('.input.', '.labels.')):
            items[r['id']]['labels'] = r['labels'].split()
    return items


def collect(args):
    items = load(Path(args.work))
    mixed = defaultdict(set)
    for r in items.values():
        for w, c in zip(r['words'], r['labels']):
            if w.islower():
                mixed[w].add(c)
    mixed = {w for w, cs in mixed.items() if {'e', 'n'} <= cs and w not in HOMOGRAPHS}
    rows = []
    for r in items.values():
        w, c = r['words'], r['labels']
        for k in range(len(w)):
            insertion = (c[k] == 'e' and 0 < k < len(w) - 1 and c[k - 1] == 'n' and c[k + 1] == 'n')
            capital = k > 0 and w[k][:1].isupper() and w[k] != 'I'
            if c[k] == 'u' or capital or insertion or w[k] in mixed:
                rows.append({'cid': len(rows) + 1, 'id': r['id'], 'k': k, 'word': w[k],
                             'current': c[k], 'text': r['text']})
    out = Path(args.out)
    if out.exists():
        raise ValueError(f'{out} exists')
    write_jsonl(out, rows)
    print(f'{len(rows)} candidates -> {out}')


def apply(args):
    """harmonize_changes.jsonl holds IDs and character offsets only, never the word itself."""
    work = Path(args.work)
    items = load(work)
    script_native = json.loads((work / 'prepare_manifest.json').read_text(encoding='utf-8'))['script_native']
    sequence = {r['id']: r['sequence_id'] for r in read_jsonl(work / 'index.jsonl')}
    decisions = {r['cid']: r['label'] for r in read_jsonl(args.decisions)}
    candidates = read_jsonl(args.candidates)
    if set(decisions) != {c['cid'] for c in candidates} or any(v not in 'neu' for v in decisions.values()):
        raise ValueError('Decisions must cover every candidate exactly, with codes n/e/u')
    changes = []
    for c in candidates:
        r = items[c['id']]
        if r['words'][c['k']] != c['word'] or r['labels'][c['k']] != c['current']:
            raise ValueError(f'Candidate {c["cid"]} does not match the work dir')
        new = decisions[c['cid']]
        if new != c['current']:
            a, z = words(r['text'], script_native)[c['k']]
            changes.append({'sequence_id': sequence[c['id']], 'word_index': c['k'], 'start': a, 'end': z,
                            'before': c['current'], 'after': new})
            r['labels'][c['k']] = new
    out = new_dir(args.out)
    for p in work.glob('batch_*.input.jsonl'):
        shutil.copy(p, out / p.name)
    for name in ['index.jsonl', 'prepare_manifest.json']:
        shutil.copy(work / name, out / name)
    by_batch = defaultdict(list)
    for r in sorted(items.values(), key=lambda r: r['id']):
        by_batch[r['batch']].append({'id': r['id'], 'labels': ' '.join(r['labels'])})
    for b, rows in by_batch.items():
        write_jsonl(out / f'{b}.labels.jsonl', rows)
    write_jsonl(out / 'harmonize_changes.jsonl', changes)
    write_json(out / 'harmonize_manifest.json', {'source_work': str(work), 'candidates': len(candidates),
               'changes': len(changes), 'candidates_sha256': digest(Path(args.candidates).read_bytes()),
               'decisions_sha256': digest(Path(args.decisions).read_bytes()),
               'script_sha256': digest(Path(__file__).read_bytes())})
    print(f'{len(changes)} of {len(candidates)} candidates changed -> {out}')


def assemble(args):
    rows = read_jsonl(args.texts)
    work = Path(args.work)
    index = {r['sequence_id']: r['id'] for r in read_jsonl(work / 'index.jsonl')}
    codes = {}
    for p in sorted(work.glob('batch_*.labels.jsonl')):
        for r in read_jsonl(p):
            if r['id'] in codes:
                raise ValueError(f'Duplicate labels for id {r["id"]}')
            codes[r['id']] = r['labels'].split()
    lids, counts, problems = [], Counter(), []
    date = datetime.date.today().isoformat()
    for row in rows:
        native = native_of(row)
        lang = {'n': native, 'e': 'en', 'u': 'UNKNOWN'}
        item_id = index[row['sequence_id']]
        spans = words(row['text'], args.script_native)
        labs = codes.get(item_id)
        if labs is None or len(labs) != len(spans) or any(c not in lang for c in labs):
            problems.append(item_id)
            continue
        if args.script_native:
            chars = [s['lang'] for s in annotate_script.segments_for(row['text'], native) for _ in s['text']]
        else:
            chars = ['NEUTRAL'] * len(row['text'])
        for (a, z), c in zip(spans, labs):
            chars[a:z] = [lang[c]] * (z - a)
        segs = []
        for ch, lab in zip(row['text'], chars):
            if segs and segs[-1]['lang'] == lab:
                segs[-1]['text'] += ch
            else:
                segs.append({'text': ch, 'lang': lab})
        lid = {'sequence_id': row['sequence_id'], 'text_sha256': row['text_sha256'],
               'annotation_source': f'{args.model}/claude-code/annotate_claude.py/prompt-{PROMPT_VERSION}/{date}',
               'review_status': 'unreviewed', 'reviewer': '', 'segments': segs}
        validate_lid(row, lid)
        lids.append(lid)
        for s in segs:
            counts[s['lang']] += len(s['text'].strip())
    if problems:
        raise ValueError(f'{len(problems)} items have missing or misaligned labels, e.g. ids {problems[:20]}')
    out = new_dir(args.out)
    write_jsonl(out / 'texts.jsonl', rows)
    write_jsonl(out / 'lid.jsonl', lids)
    write_json(out / 'lid_manifest.json', {'method': 'claude-word-level-lid', 'model': args.model,
               'interface': 'Claude Code (interactive session, subagents)',
               'prompt': f'claude_lid_instructions.md (LID_PROMPT.md {PROMPT_VERSION}, applied per word)',
               'script_native': args.script_native,
               'date': date, 'input_texts': args.texts, 'sequences': len(lids),
               'non_space_chars_by_label': dict(counts),
               'labels_sha256': {p.name: digest(p.read_bytes()) for p in sorted(work.glob('batch_*.labels.jsonl'))},
               'script_sha256': digest(Path(__file__).read_bytes())})
    print(f'Labeled {len(lids)} sequences; chars by label {dict(counts)}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('prepare')
    s.add_argument('--texts', required=True)
    s.add_argument('--batch-size', type=int, default=100)
    s.add_argument('--script-native', action='store_true',
                   help='Label native-script text by annotate_script.py; send only all-Latin words to the labeler')
    s.add_argument('--out', required=True)
    s.set_defaults(func=prepare)
    s = sub.add_parser('collect')
    s.add_argument('--work', required=True)
    s.add_argument('--out', required=True)
    s.set_defaults(func=collect)
    s = sub.add_parser('apply')
    s.add_argument('--work', required=True)
    s.add_argument('--candidates', required=True)
    s.add_argument('--decisions', required=True, help='JSONL of {"cid": n, "label": "n|e|u"}')
    s.add_argument('--out', required=True)
    s.set_defaults(func=apply)
    s = sub.add_parser('assemble')
    s.add_argument('--texts', required=True)
    s.add_argument('--work', required=True, help='Directory from prepare (or apply), with batch_*.labels.jsonl')
    s.add_argument('--model', default='claude-opus-5-5')
    s.add_argument('--script-native', action='store_true', help='Must match the prepare step')
    s.add_argument('--out', required=True)
    s.set_defaults(func=assemble)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
