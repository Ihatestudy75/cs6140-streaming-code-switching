#!/usr/bin/env python3
"""Reproduce Yufan's four-pair data preparation. No Git operations.

Run from the repository root (Python 3.13 recommended, 3.11+ required):
    python3 project_update_1/run_lingua.py --setup-only
    python3 project_update_1/run_lingua.py

The first command creates project_update_1/.venv if absent, installs the
shared pinned requirements plus Lingua, and downloads only the frozen tokenizer.
The second builds deliverables/yufan. It REFUSES to overwrite an existing output.
Use --out deliverables/yufan_reproduction to retain the current delivery.

Offline reuse after setup:
    python3 project_update_1/run_lingua.py --skip-install
Read-only verification (no package installation or network):
    python3 project_update_1/run_lingua.py --verify deliverables/yufan
Small end-to-end check, separate from the 500-record delivery:
    python3 project_update_1/run_lingua.py --skip-install \
        --extract-limit 50 --records 5 --work-dir project_update_1/data/yufan_smoke \
        --out project_update_1/data/yufan_smoke_delivery

Required repository files: this file, pipeline.py, annotate_lingua.py, and
requirements.lock.txt in project_update_1/. Original CSV files belong in
datasets/ at the repository root. No API key, GPU, PyTorch,
PinoyBot, Git LFS, Excel, or separate Lingua model download is required.

Existing source code/data are preserved. Full working and quarantined data stay
under --work-dir. Only sampled retained data enter --out. Completed stages are
reused only when their input and output checksums match. Interrupted incomplete
stages require a new --work-dir; nothing is silently deleted.

All LID is UNREVIEWED. Screening is a conservative heuristic, not human review.
The 58 historical French disagreement decisions embedded below preserve the
original development exclusions. They are not a gold set and apply only to the
original CSV IDs. The script computes actual counts, never forces 4,908 rows.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import subprocess
import sys
import venv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PAIRS = {
    'fr_en': ('French_eng.csv', 'fr'),
    'de_en': ('German_Eng.csv', 'de'),
    'it_en': ('Italian_eng.csv', 'it'),
    'philippines_en': ('Philippines_eng.csv', 'philippines'),
}
MODEL = 'google-bert/bert-base-multilingual-cased'
REVISION = '3f076fdb1ab68d5b2880cb87a0886f315b8146f8'
TOKENIZER_HASH = '78b44c32238e3b5e83d1deaa63c3fa1466ddf73547ce641a36f75180079ab45f'
PHRASES = {
    'you know': r'\byou know\b', 'right?': r'\bright\?',
    'for sure': r'\bfor sure\b', "don't you think": r'\bdon[’\x27]t you think\b',
    'honestly': r'\bhonestly\b', 'trust me': r'\btrust me\b',
    'you see': r'\byou see\b', 'anyway': r'\banyway\b',
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, values):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + '\n')


def repository(explicit=None):
    if explicit:
        candidates = [Path(explicit).resolve()]
    else:
        candidates = [HERE, *HERE.parents, Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / 'project_update_1/pipeline.py').is_file():
            return candidate
    raise ValueError('Cannot find repository. Pass --repo-root with project_update_1/.')


def relative(root, value):
    p = Path(value)
    return (p if p.is_absolute() else root / p).resolve()


def stage(folder, key, action):
    """Checkpoint whole stages, without reusing changed/partial results."""
    marker = folder / '_completed_yufan.json'
    if folder.exists():
        check(marker.is_file(), f'Incomplete stage at {folder}. Choose a new --work-dir.')
        previous = load(marker)
        check(previous['key'] == key, f'Changed inputs/configuration: {folder}. Choose a new --work-dir.')
        actual = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*')
                  if p.is_file() and p != marker}
        check(actual == previous['files'], f'Changed stage files: {folder}')
        print(f'Reuse verified stage: {folder.name}', flush=True)
        return
    action()
    check(folder.is_dir(), f'Stage did not create {folder}')
    save(marker, {'key': key, 'files': {str(p.relative_to(folder)): sha(p)
                                      for p in folder.rglob('*') if p.is_file()}})


def screening_reasons(pair, text, lid, token):
    chars = [s['lang'] for s in lid['segments'] for _ in s['text']]
    check(len(chars) == len(text['text']), 'LID reconstruction length mismatch')
    reasons = []
    for pattern in PHRASES.values():
        for match in re.finditer(pattern, text['text'], re.I):
            langs = {chars[i] for i in range(match.start(), match.end()) if text['text'][i].isalpha()}
            if langs != {'en'}:
                reasons.append('short_english_phrase_not_labeled_en')
    if any(c.isalpha() and chars[i] == 'NEUTRAL' for i, c in enumerate(text['text'])):
        reasons.append('alphabetic_neutral')
    if 'UNKNOWN' in chars:
        reasons.append('unknown_lid')
    if 'UNKNOWN' in token['language_ids'] or any(
            n and n != 'neutral_inherited_previous' for n in token['alignment_notes']):
        reasons.append('unknown_or_ambiguous_subword')
    if any(sw == 1 and dur == -1 for sw, dur in zip(token['switch_labels'], token['duration_labels'])):
        reasons.append('switch_without_duration')
    if not any(token['switch_valid_mask']):
        reasons.append('no_valid_prediction_position')
    if pair == 'fr_en' and text['sequence_id'] in HISTORICAL_FRENCH_DISAGREEMENTS:
        reasons.append('disagreement_with_existing_french_preannotation')
    if pair == 'philippines_en':
        for match in re.finditer(r'\bat\b', text['text'], re.I):
            if {chars[i] for i in range(match.start(), match.end())} == {'en'}:
                reasons.append('english_labeled_at_requires_context_review')
    return sorted(set(reasons))


def screen(pair, extraction, annotation, tokenized, output, core):
    output.mkdir(parents=True)
    texts, lids, tokens = rows(annotation / 'texts.jsonl'), rows(annotation / 'lid.jsonl'), rows(tokenized / 'labeled_tokens.jsonl')
    check(len(texts) == len(lids) == len(tokens), 'Mismatched stage lengths')
    issue_ids = {r['id'] for r in rows(extraction / 'extract_issues.jsonl')}
    partitions = {'retained': [], 'quarantined': []}
    reasons = []
    for text, lid, token in zip(texts, lids, tokens):
        check(text['sequence_id'] == lid['sequence_id'] == token['sequence_id'], 'Mismatched IDs')
        core.validate_lid(text, lid)
        core.validate_row(token)
        flags = screening_reasons(pair, text, lid, token)
        if text['sequence_id'] in issue_ids or text['record_id'] in issue_ids:
            flags.append('extraction_issue')
        group = 'quarantined' if flags else 'retained'
        partitions[group].append((text, lid, token))
        if flags:
            reasons.append({'sequence_id': text['sequence_id'], 'record_id': text['record_id'],
                            'reasons': sorted(set(flags)), 'has_subword_labels': True})
    for group, data in partitions.items():
        folder = output / group
        folder.mkdir()
        for index, name in enumerate(['texts.jsonl', 'lid.jsonl', 'labeled_tokens.jsonl']):
            write_rows(folder / name, [r[index] for r in data])
        save(folder / 'stats.json', core.summary([r[2] for r in data]))
    write_rows(output / 'quarantine_reasons.jsonl', reasons)
    text_map = {t['sequence_id']: t for t in texts}
    with (output / 'review.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sequence_id', 'reviewer', 'review_date', 'status', 'issue_type', 'before', 'after', 'reason'])
        for item in reasons:
            writer.writerow([item['sequence_id'], '', '', 'needs_review', ';'.join(item['reasons']),
                             text_map[item['sequence_id']]['text'], '', 'Automatic review candidate, not a confirmed linguistic error.'])
    save(output / 'screening_manifest.json', {
        'input_sequences': len(texts), 'retained': len(partitions['retained']),
        'quarantined': len(partitions['quarantined']),
        'pre_lid_exclusions': load(annotation / 'lid_manifest.json'),
        'reason_counts': dict(Counter(r for item in reasons for r in item['reasons'])),
        'english_phrase_patterns': PHRASES,
        'at_rule': 'Philippines only: standalone at labeled en is a context-review candidate, not necessarily wrong.',
        'french_history': '58 frozen unreviewed pilot-disagreement exclusions, preserved by original source ID.',
        'scope': 'Whole sequences. Text and labels are unchanged. All outputs remain unreviewed.',
        'limitations': ['Unflagged LID errors can remain.', 'Filtering can bias short-switch and duration distributions.',
                        'Quoted speech is eligible; extraction warnings are deferred, not permanent deletion.']})
    print(f'{pair}: retained {len(partitions["retained"])}; quarantined {len(reasons)}', flush=True)


def raw_metadata(raw_file, texts):
    with raw_file.open(encoding='utf-8-sig', newline='') as handle:
        source = list(csv.DictReader(handle))
    checksum = sha(raw_file)
    parsed, result = {}, []
    for text in texts:
        number = text['source_row']
        original = source[number - 1]
        check(text['source_file'] == raw_file.name, 'Source filename mismatch')
        check(text['record_id'] == f'{text["language_pair"]}:{checksum[:12]}:{number:06d}', 'Source ID mismatch')
        if number not in parsed:
            try:
                parsed[number] = json.loads(original['data_generation_result'])
            except json.JSONDecodeError:
                parsed[number] = ast.literal_eval(original['data_generation_result'])
        value = parsed[number]
        for step in text['source_path']:
            value = value[step]
        check(value == text['text'], 'Raw text mismatch')
        check(original['cs_type'] in {'Intersentential', 'Intrasentential', 'Extra-sentential / Tag switching'}, 'Unexpected cs_type')
        result.append({'sequence_id': text['sequence_id'], 'record_id': text['record_id'],
                       'source_row': number, 'cs_type': original['cs_type']})
    return result


def package(root, work, destination, pairs, records, seed, core, args):
    check(not destination.exists(), f'Output already exists: {destination}')
    staging = work / 'delivery_stage'
    check(not staging.exists(), f'Unpublished staging output exists: {staging}. Choose a new --work-dir.')
    staging.mkdir()
    totals = {}
    for pair in pairs:
        base = work / pair
        retained = base / 'screened/retained'
        texts, lids, tokens = (rows(retained / name) for name in ['texts.jsonl', 'lid.jsonl', 'labeled_tokens.jsonl'])
        record_ids = sorted({r['record_id'] for r in texts})
        check(len(record_ids) >= records, f'{pair}: only {len(record_ids)} retained records, fewer than --records {records}')
        selected = set(random.Random(f'{seed}:{pair}').sample(record_ids, records))
        triples = [(t, l, k) for t, l, k in zip(texts, lids, tokens) if t['record_id'] in selected]
        chosen_texts, chosen_lids, chosen_tokens = ([r[i] for r in triples] for i in range(3))
        ids = {r['sequence_id'] for r in chosen_texts}
        folder = staging / pair
        for sub in ['extraction', 'annotation', 'tokenized']:
            (folder / sub).mkdir(parents=True)
        write_rows(folder / 'extraction/texts_yufan.jsonl', chosen_texts)
        templates = [r for r in rows(base / 'extraction/lid.template.jsonl') if r['sequence_id'] in ids]
        write_rows(folder / 'extraction/lid.template_yufan.jsonl', templates)
        shutil.copy2(base / 'extraction/extract_issues.jsonl', folder / 'extraction/extract_issues_yufan.jsonl')
        save(folder / 'extraction/extract_manifest_yufan.json', {
            'full_extraction': load(base / 'extraction/extract_manifest.json'), 'delivered_sequences': len(ids),
            'source_text_verified_against_csv': True, 'issues_scope': 'Full extraction, not only selected sequences'})
        save(folder / 'extraction/sampling_manifest_yufan.json', {
            'method': 'random sample without replacement from sorted retained record_ids',
            'seed_expression': f'{seed}:{pair}', 'sampled_records': records,
            'eligible_records': len(record_ids), 'source_retained_sequences': len(texts),
            'sampled_sequences': len(ids), 'selected_record_ids': sorted(selected),
            'all_existing_retained_items_of_selected_records_included': True,
            'sampling_population': 'screened retained set, NOT entire raw CSV', 'labels_modified': False})
        write_rows(folder / 'annotation/lid_yufan.jsonl', chosen_lids)
        save(folder / 'annotation/lid_manifest_yufan.json', {
            'source_annotation_run': load(base / 'annotation/lid_manifest.json'),
            'delivered_sequences': len(ids), 'independently_verified': False})
        shutil.copy2(base / 'screened/screening_manifest.json', folder / 'annotation/screening_manifest_yufan.json')
        q = [r for r in rows(base / 'screened/quarantine_reasons.jsonl') if r['record_id'] in selected]
        write_rows(folder / 'annotation/quarantine_reasons_yufan.jsonl', q)
        with (base / 'screened/review.csv').open(encoding='utf-8', newline='') as src, \
                (folder / 'annotation/review_yufan.csv').open('x', encoding='utf-8', newline='') as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(r for r in reader if r['sequence_id'].rsplit(':', 1)[0] in selected)
        development = sorted(selected & set(HISTORICAL_DEVELOPMENT_RECORDS.get(pair, [])))
        (folder / 'annotation/pilot_record_ids_yufan.txt').write_text(''.join(r + '\n' for r in development), encoding='utf-8')
        save(folder / 'annotation/development_record_manifest_yufan.json', {
            'present_in_delivery': development,
            'meaning': 'Previously inspected development records, NOT independently reviewed gold labels.'})
        write_rows(folder / 'tokenized/labeled_tokens_yufan.jsonl', chosen_tokens)
        write_rows(folder / 'tokenized/token_issues_yufan.jsonl',
                   [r for r in rows(base / 'tokenized/token_issues.jsonl') if r['sequence_id'] in ids])
        stats = core.summary(chosen_tokens)
        save(folder / 'tokenized/stats_yufan.json', stats)
        metadata = raw_metadata(root / 'datasets' / PAIRS[pair][0], chosen_texts)
        write_rows(folder / 'cs_type_yufan.jsonl', metadata)
        save(folder / 'tokenized/manifest_yufan.json', {
            'source_tokenization_run': load(base / 'tokenized/manifest.json'),
            'neutral_policy': 'inherit_previous', 'end_policy': 'observed',
            'review_status': 'unreviewed', 'independent_linguistic_verification': False,
            'cs_type_note': 'Original-record metadata, not guaranteed to describe each extracted entry.',
            'output_sha256': {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*.jsonl')}})
        totals[pair] = {'records': records, 'sequences': len(ids), 'stats': stats,
                        'source_eligible_records': len(record_ids), 'source_retained_sequences': len(texts),
                        'cs_type_counts': dict(Counter(r['cs_type'] for r in metadata))}
        print(f'Delivery {pair}: {records} records, {len(ids)} sequences', flush=True)
    save(staging / 'summary_yufan.json', totals)
    scriptdir = staging / 'scripts'
    scriptdir.mkdir()
    for source in [Path(__file__).resolve(), root / 'project_update_1/pipeline.py',
                   root / 'project_update_1/annotate_lingua.py']:
        shutil.copy2(source, scriptdir / source.name)
    lock = root / 'project_update_1/requirements.lock.txt'
    (staging / 'requirements_yufan.txt').write_text(lock.read_text(encoding='utf-8').rstrip() + '\nlingua-language-detector==2.2.0\n', encoding='utf-8')
    save(staging / 'runtime_yufan.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'versions': {p: importlib.metadata.version(p) for p in ['transformers', 'tokenizers', 'lingua-language-detector']},
        'tokenizer': load(root / 'project_update_1/assets/mbert/tokenizer_manifest.json'),
        'code_sha256': {p.name: sha(p) for p in scriptdir.iterdir()},
        'raw_sha256': {pair: sha(root / 'datasets' / PAIRS[pair][0]) for pair in pairs},
        'sampling_seed': seed, 'records_per_pair': records, 'extraction_record_limit': args.extract_limit,
        'linguistically_verified': False, 'no_git_operations': True})
    (staging / 'README.md').write_text(
        '# Yufan data processing delivery\n\n'
        'Automatic Lingua span LID, frozen mBERT alignment, switch/duration labels. All LID remains unreviewed.\n\n'
        'Run from repository root (requires project_update_1/ and datasets/):\n\n'
        '```bash\npython3 project_update_1/run_lingua.py --setup-only\n'
        'python3 project_update_1/run_lingua.py\n```\n\n'
        'Existing outputs are never overwritten. For a second run pass --out deliverables/yufan_reproduction.\n'
        'Read-only verification: python3 project_update_1/run_lingua.py --verify <delivery-path>.\n\n'
        'Sampling selects records from the screened pool and keeps all remaining sequences of selected records.\n'
        'Extraction issues are deferred before LID, exact text duplicates are removed, and whole suspicious sequences are quarantined.\n'
        'French historical disagreement IDs are frozen development decisions, not human-verified truth.\n'
        'Philippines source labels use Lingua TAGALOG based on SwitchLingua naming Filipino (Tagalog).\n'
        'English-labeled standalone at is a review candidate, not an automatically confirmed error.\n'
        'Neutral punctuation inherits the preceding language and counts toward duration. Text ends are observed boundaries.\n'
        'Full text is for target construction only. Prediction inputs must exclude future text and answer metadata.\n'
        'Raw files, full intermediates, quarantined data, tokenizer assets, and the virtual environment stay outside this delivery.\n', encoding='utf-8')
    save(staging / 'file_inventory_yufan.json', {'files': [
        {'path': str(p.relative_to(staging)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(staging.rglob('*')) if p.is_file()]})
    verify(staging, core)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publication within the same repository; never replace existing outputs.
    check(not destination.exists(), f'Output appeared during processing: {destination}')
    staging.rename(destination)
    print(f'COMPLETE: {destination}. No Git operations.', flush=True)


def verify(destination, core):
    inventory = load(destination / 'file_inventory_yufan.json')['files']
    for item in inventory:
        p = destination / item['path']
        check(p.is_file() and sha(p) == item['sha256'] and p.stat().st_size == item['bytes'], f'Checksum mismatch: {p}')
    totals = load(destination / 'summary_yufan.json')
    for pair, expected in totals.items():
        folder = destination / pair
        groups = [rows(folder / f) for f in ['extraction/texts_yufan.jsonl', 'annotation/lid_yufan.jsonl',
                  'tokenized/labeled_tokens_yufan.jsonl', 'cs_type_yufan.jsonl', 'extraction/lid.template_yufan.jsonl']]
        check(len({len(g) for g in groups}) == 1 and len(groups[0]) == expected['sequences'], 'Delivery lengths mismatch')
        check(len({r['sequence_id'] for r in groups[0]}) == len(groups[0]), 'Duplicate sequence IDs')
        for text, lid, token, cs_type, template in zip(*groups):
            check(len({r['sequence_id'] for r in [text, lid, token, cs_type, template]}) == 1, 'Delivery IDs mismatch')
            check(text['text'] == token['text'] and text['record_id'] == token['record_id'] == cs_type['record_id'], 'Delivery text/record mismatch')
            check(lid['review_status'] == token['review_status'] == 'unreviewed', 'Unexpected review status')
            core.validate_lid(text, lid)
            core.validate_row(token)
        check(core.summary(groups[2]) == load(folder / 'tokenized/stats_yufan.json') == expected['stats'], 'Statistics mismatch')
        check(expected['stats']['switches_without_duration'] == 0 and expected['stats']['unknown_subwords'] == 0, 'Unexpected unscreened target')
        sample = load(folder / 'extraction/sampling_manifest_yufan.json')
        record_ids = {r['record_id'] for r in groups[0]}
        check(record_ids == set(sample['selected_record_ids']) and len(record_ids) == expected['records'], 'Record sample mismatch')
        check(len(groups[0]) == sample['sampled_sequences'], 'Sampling count mismatch')
        print(f'PASS {pair}: {len(groups[0])} sequences; structure/provenance only, NOT LID accuracy.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root')
    parser.add_argument('--venv', default='project_update_1/.venv')
    parser.add_argument('--work-dir', default='project_update_1/data/yufan_reproduction')
    parser.add_argument('--out', default='deliverables/yufan')
    parser.add_argument('--pairs', nargs='+', choices=list(PAIRS), default=list(PAIRS))
    parser.add_argument('--records', type=int, default=500)
    parser.add_argument('--seed', type=int, default=6140)
    parser.add_argument('--extract-limit', type=int, default=0, help='Source CSV records per pair; 0 means all. Use only for smoke checks.')
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--skip-install', action='store_true', help='Reuse installed pinned dependencies and cached tokenizer')
    parser.add_argument('--verify', metavar='DELIVERY', help='Read-only verification without environment setup')
    args = parser.parse_args()
    check(sys.version_info >= (3, 11), 'Python 3.11+ is required. Python 3.13 is recommended.')
    check(args.records > 0 and args.extract_limit >= 0, 'Invalid sample/limit argument')
    check(len(args.pairs) == len(set(args.pairs)), 'Duplicate language pair')
    root = repository(args.repo_root)
    standard = root / 'project_update_1'
    sys.path.insert(0, str(standard))
    import pipeline as core
    if args.verify:
        verify(relative(root, args.verify), core)
        return
    destination, work = relative(root, args.out), relative(root, args.work_dir)
    if not args.setup_only:
        check(not destination.exists(), f'{destination} already exists. Preserve it and pass --out deliverables/yufan_reproduction.')
        check(not destination.is_relative_to(work) and not work.is_relative_to(destination), '--out and --work-dir must be separate directories')
        for pair in args.pairs:
            check((root / 'datasets' / PAIRS[pair][0]).is_file(), f'Missing source CSV: {PAIRS[pair][0]}')
    env = relative(root, args.venv)
    executable = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    inside = Path(sys.prefix).resolve() == env.resolve()
    if not inside:
        if not executable.exists():
            check(not args.skip_install, '--skip-install requires an existing virtual environment')
            print(f'Create Python environment: {env}', flush=True)
            venv.EnvBuilder(with_pip=True).create(env)
        if not args.skip_install:
            subprocess.run([str(executable), '-m', 'pip', 'install', '-r', str(standard / 'requirements.lock.txt'),
                            'lingua-language-detector==2.2.0'], check=True)
        command = [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]]
        if '--skip-install' not in command:
            command.append('--skip-install')
        child_env = dict(os.environ)
        if not args.skip_install:
            child_env['YUFAN_ALLOW_TOKENIZER_DOWNLOAD'] = '1'
        subprocess.run(command, check=True, env=child_env)
        return
    if not args.skip_install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(standard / 'requirements.lock.txt'),
                        'lingua-language-detector==2.2.0'], check=True)
    for package_name, expected in [('lingua-language-detector', '2.2.0'), ('transformers', '4.57.6'), ('tokenizers', '0.22.2')]:
        check(importlib.metadata.version(package_name) == expected, f'{package_name} must be {expected}; run setup with the system Python.')
    assets = standard / 'assets/mbert'
    if not assets.exists():
        check(not args.skip_install or os.environ.get('YUFAN_ALLOW_TOKENIZER_DOWNLOAD') == '1',
              'Tokenizer missing. Run --setup-only without --skip-install to download it.')
        core.prepare_tokenizer(SimpleNamespace(model=MODEL, revision=REVISION, out=str(assets)))
    manifest = load(assets / 'tokenizer_manifest.json')
    check(manifest['model'] == MODEL and manifest['revision'] == REVISION,
          'Existing tokenizer model/revision differs from the frozen team asset')
    check(core.tree_hash(assets / 'tokenizer') == manifest['tokenizer_sha256'] == TOKENIZER_HASH,
          'Frozen tokenizer checksum mismatch. Preserve asset and inspect it.')
    if args.setup_only:
        print('SETUP COMPLETE: pinned dependencies and frozen tokenizer ready.', flush=True)
        return
    work.mkdir(parents=True, exist_ok=True)
    code_hashes = {name: sha(standard / name) for name in ['pipeline.py', 'annotate_lingua.py']}
    code_hashes['entrypoint'] = sha(Path(__file__))
    for pair in args.pairs:
        filename, language = PAIRS[pair]
        base = work / pair
        extraction, annotation, tokenized, screened = [base / n for n in ['extraction', 'annotation', 'tokenized', 'screened']]
        base.mkdir(parents=True, exist_ok=True)
        raw = root / 'datasets' / filename
        key = {'code': code_hashes, 'raw': sha(raw), 'pair': pair, 'limit': args.extract_limit,
               'tokenizer': manifest, 'lingua': '2.2.0', 'policies': ['inherit_previous', 'observed']}
        stage(extraction, key, lambda: core.extract(SimpleNamespace(input=str(raw), languages=f'{language},en',
                                                                 limit=args.extract_limit, out=str(extraction))))
        stage(annotation, key, lambda: subprocess.run([
            sys.executable, str(standard / 'annotate_lingua.py'), '--texts', str(extraction / 'texts.jsonl'),
            '--issues', str(extraction / 'extract_issues.jsonl'), '--source-language', language,
            '--limit', '0', '--out', str(annotation)], check=True))
        stage(tokenized, key, lambda: core.tokenize(SimpleNamespace(texts=str(annotation / 'texts.jsonl'),
              lid=str(annotation / 'lid.jsonl'), tokenizer_dir=str(assets), neutral_policy='inherit_previous',
              end_policy='observed', allow_unreviewed=True, out=str(tokenized))))
        stage(screened, key, lambda: screen(pair, extraction, annotation, tokenized, screened, core))
    package(root, work, destination, args.pairs, args.records, args.seed, core, args)


# Frozen historical exclusions and previously inspected record IDs follow below.
# These are development provenance only. No reviewer or gold status is invented.
HISTORICAL_FRENCH_DISAGREEMENTS = ['fr_en:62970a8adc63:000001:001',
 'fr_en:62970a8adc63:000002:001',
 'fr_en:62970a8adc63:000003:001',
 'fr_en:62970a8adc63:000005:001',
 'fr_en:62970a8adc63:000006:001',
 'fr_en:62970a8adc63:000008:001',
 'fr_en:62970a8adc63:000009:001',
 'fr_en:62970a8adc63:000010:001',
 'fr_en:62970a8adc63:000011:001',
 'fr_en:62970a8adc63:000012:001',
 'fr_en:62970a8adc63:000013:003',
 'fr_en:62970a8adc63:000013:004',
 'fr_en:62970a8adc63:000014:002',
 'fr_en:62970a8adc63:000015:001',
 'fr_en:62970a8adc63:000015:002',
 'fr_en:62970a8adc63:000016:002',
 'fr_en:62970a8adc63:000017:001',
 'fr_en:62970a8adc63:000019:002',
 'fr_en:62970a8adc63:000019:003',
 'fr_en:62970a8adc63:000020:001',
 'fr_en:62970a8adc63:000020:003',
 'fr_en:62970a8adc63:000021:001',
 'fr_en:62970a8adc63:000021:002',
 'fr_en:62970a8adc63:000021:004',
 'fr_en:62970a8adc63:000024:001',
 'fr_en:62970a8adc63:000024:002',
 'fr_en:62970a8adc63:000024:003',
 'fr_en:62970a8adc63:000024:004',
 'fr_en:62970a8adc63:000025:001',
 'fr_en:62970a8adc63:000025:002',
 'fr_en:62970a8adc63:000025:003',
 'fr_en:62970a8adc63:000025:004',
 'fr_en:62970a8adc63:000025:005',
 'fr_en:62970a8adc63:000027:001',
 'fr_en:62970a8adc63:000027:002',
 'fr_en:62970a8adc63:000027:003',
 'fr_en:62970a8adc63:000027:004',
 'fr_en:62970a8adc63:000028:001',
 'fr_en:62970a8adc63:000028:002',
 'fr_en:62970a8adc63:000029:001',
 'fr_en:62970a8adc63:000029:002',
 'fr_en:62970a8adc63:000029:003',
 'fr_en:62970a8adc63:000029:004',
 'fr_en:62970a8adc63:000030:001',
 'fr_en:62970a8adc63:000031:001',
 'fr_en:62970a8adc63:000031:002',
 'fr_en:62970a8adc63:000031:003',
 'fr_en:62970a8adc63:000031:004',
 'fr_en:62970a8adc63:000033:002',
 'fr_en:62970a8adc63:000034:002',
 'fr_en:62970a8adc63:000034:003',
 'fr_en:62970a8adc63:000035:001',
 'fr_en:62970a8adc63:000037:001',
 'fr_en:62970a8adc63:000037:002',
 'fr_en:62970a8adc63:000037:003',
 'fr_en:62970a8adc63:000037:004',
 'fr_en:62970a8adc63:000038:002',
 'fr_en:62970a8adc63:000038:004']
HISTORICAL_DEVELOPMENT_RECORDS = {'de_en': ['de_en:8e3f9a07c453:000003',
           'de_en:8e3f9a07c453:000005',
           'de_en:8e3f9a07c453:000006',
           'de_en:8e3f9a07c453:000008',
           'de_en:8e3f9a07c453:000260',
           'de_en:8e3f9a07c453:002224',
           'de_en:8e3f9a07c453:002432',
           'de_en:8e3f9a07c453:005321',
           'de_en:8e3f9a07c453:006114',
           'de_en:8e3f9a07c453:006144'],
 'fr_en': ['fr_en:62970a8adc63:000001',
           'fr_en:62970a8adc63:000002',
           'fr_en:62970a8adc63:000003',
           'fr_en:62970a8adc63:000005',
           'fr_en:62970a8adc63:000006',
           'fr_en:62970a8adc63:000007',
           'fr_en:62970a8adc63:000008',
           'fr_en:62970a8adc63:000009',
           'fr_en:62970a8adc63:000010',
           'fr_en:62970a8adc63:000011',
           'fr_en:62970a8adc63:000012',
           'fr_en:62970a8adc63:000013',
           'fr_en:62970a8adc63:000014',
           'fr_en:62970a8adc63:000015',
           'fr_en:62970a8adc63:000016',
           'fr_en:62970a8adc63:000017',
           'fr_en:62970a8adc63:000018',
           'fr_en:62970a8adc63:000019',
           'fr_en:62970a8adc63:000020',
           'fr_en:62970a8adc63:000021',
           'fr_en:62970a8adc63:000022',
           'fr_en:62970a8adc63:000023',
           'fr_en:62970a8adc63:000024',
           'fr_en:62970a8adc63:000025',
           'fr_en:62970a8adc63:000026',
           'fr_en:62970a8adc63:000027',
           'fr_en:62970a8adc63:000028',
           'fr_en:62970a8adc63:000029',
           'fr_en:62970a8adc63:000030',
           'fr_en:62970a8adc63:000031',
           'fr_en:62970a8adc63:000032',
           'fr_en:62970a8adc63:000033',
           'fr_en:62970a8adc63:000034',
           'fr_en:62970a8adc63:000035',
           'fr_en:62970a8adc63:000036',
           'fr_en:62970a8adc63:000037',
           'fr_en:62970a8adc63:000038'],
 'it_en': ['it_en:d76eac6442a1:000001',
           'it_en:d76eac6442a1:000005',
           'it_en:d76eac6442a1:000244',
           'it_en:d76eac6442a1:002121',
           'it_en:d76eac6442a1:002327',
           'it_en:d76eac6442a1:005062',
           'it_en:d76eac6442a1:005796',
           'it_en:d76eac6442a1:005818'],
 'philippines_en': ['philippines_en:b111cae514ec:000008']}

if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, importlib.metadata.PackageNotFoundError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
