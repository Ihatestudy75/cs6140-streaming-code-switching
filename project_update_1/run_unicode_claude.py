#!/usr/bin/env python3
"""Reproduce yulun's six-pair data preparation: ar_en, zh_en, ja_en, hi_en, es_en, vi_en. No Git operations.

Run from the repository root (Python 3.11+):
    python project_update_1/run_unicode_claude.py --setup-only
    python project_update_1/run_unicode_claude.py --skip-install

The first command creates project_update_1/.venv if absent, installs the shared pinned
requirements, and downloads only the frozen mBERT tokenizer. The second builds the full delivery
in deliverables/yulun_full (ignored by Git; it contains dataset text) and compares its
text-free files with the committed deliverables/yulun. It REFUSES to overwrite an existing output.

The SwitchLingua licence forbids redistribution of the dataset, in whole or in part. So
deliverables/yulun in Git keeps only files without dataset text: statistics, manifests, IDs,
cs_type, review records, and the Claude labels as character offsets.

Steps per pair: pipeline.py extract on the full CSV -> random sample of 500 records
(seed '6140:<pair>') -> LID -> review of 20 pilot records (seed '6140:<pair>:pilot')
-> pipeline.py tokenize (inherit_previous, observed) -> cs_type map -> delivery files.

LID: ar_en, zh_en, ja_en use annotate_script.py (Unicode script rule, no network).
es_en, vi_en and hi_en were labeled word by word by Claude in Claude Code (annotate_claude.py,
claude_lid_instructions.md). That cannot run unattended and its output can differ between runs,
so the runner rebuilds the delivered labels from lid_offsets_yulun.jsonl and the CSV text.

The original CSV files are not in the repository. Download them from the Hugging Face dataset
Shelton1013/SwitchLingua_text and put them in datasets/ at the repository root.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
# pair -> (CSV file in datasets/, language code, CSV sha256)
PAIRS = {
    'ar_en': ('Arabic_eng.csv', 'ar', '608a99e61cd320a2c5689158f5da1bb681d38d42c7cd29a5b7bf1cc0e833c8bd'),
    'zh_en': ('Chinese_eng.csv', 'zh', '0501713457739406055dc4b1f41fd16d424e45e8aeadf4dfb21a11939c24e225'),
    'ja_en': ('Japanese_eng.csv', 'ja', 'cbed91aed45810bc637e5f82f4db4acde4db8c0deec67098855983320cdb1cd0'),
    'hi_en': ('Hindi_eng.csv', 'hi', 'c9491f7b14b09e8b72d971b3bb85cfcdf47856f90d5fb0cb5a9ef72b07984c7c'),
    'es_en': ('Spanish_eng.csv', 'es', '9d85d88132cd483783122033a79e719edc1dfa643f11a7e0af0a40104d1a6d5d'),
    'vi_en': ('Vietnamese_eng.csv', 'vi', '26e5d66e65bb7a04351fa521110378de3aee5a99c6b3dc7d999912c3a0f0c8b7'),
}
CLAUDE_PAIRS = {'hi_en', 'es_en', 'vi_en'}
MODEL = 'google-bert/bert-base-multilingual-cased'
REVISION = '3f076fdb1ab68d5b2880cb87a0886f315b8146f8'
TOKENIZER_HASH = '78b44c32238e3b5e83d1deaa63c3fa1466ddf73547ce641a36f75180079ab45f'
SUFFIX = '_yulun'

# Model-assisted review of the 20 pilot records per pair (not an independent human review).
# It found no errors, so there are no review fixes.
REVIEWER = 'claude-opus-5-5 (model-assisted review, delegated by yulun)'
REVIEW_DATE = '2026-10-18'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def repository(explicit=None):
    candidates = [Path(explicit).resolve()] if explicit else [HERE, *HERE.parents, Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / 'project_update_1/pipeline.py').is_file():
            return candidate
    raise ValueError('Cannot find repository. Pass --repo-root with project_update_1/.')


def relative(root, value):
    p = Path(value)
    return (p if p.is_absolute() else root / p).resolve()


def prepare_tokenizer(core, assets):
    if not assets.exists():
        core.prepare_tokenizer(SimpleNamespace(model=MODEL, revision=REVISION, out=str(assets)))
        # Windows writes the tokenizer text files with CRLF; the frozen hash is for LF files.
        changed = False
        for p in (assets / 'tokenizer').rglob('*'):
            data = p.read_bytes() if p.is_file() else b''
            if b'\r\n' in data:
                p.write_bytes(data.replace(b'\r\n', b'\n'))
                changed = True
        if changed:
            manifest = json.loads((assets / 'tokenizer_manifest.json').read_text(encoding='utf-8'))
            core.write_json(assets / 'tokenizer_manifest.json',
                            dict(manifest, tokenizer_sha256=core.tree_hash(assets / 'tokenizer')))
    manifest = json.loads((assets / 'tokenizer_manifest.json').read_text(encoding='utf-8'))
    check(manifest['model'] == MODEL and manifest['revision'] == REVISION,
          f'Tokenizer in {assets} is not the frozen team tokenizer')
    check(core.tree_hash(assets / 'tokenizer') == manifest['tokenizer_sha256'] == TOKENIZER_HASH,
          f'Frozen tokenizer checksum mismatch in {assets}')


def sample(core, full, out, pair, seed, n):
    """Keep n random original records; record_id keeps the full-file row number."""
    manifest = json.loads((full / 'extract_manifest.json').read_text(encoding='utf-8'))
    total = manifest['input_records']
    numbers = sorted(random.Random(f'{seed}:{pair}').sample(range(1, total + 1), min(n, total)))
    rids = {f'{pair}:{manifest["source_sha256"][:12]}:{r:06d}' for r in numbers}
    keep = lambda sid: sid.rsplit(':', 1)[0] in rids or sid in rids
    texts = [r for r in core.read_jsonl(full / 'texts.jsonl') if r['record_id'] in rids]
    template = [r for r in core.read_jsonl(full / 'lid.template.jsonl') if keep(r['sequence_id'])]
    issues = [i for i in core.read_jsonl(full / 'extract_issues.jsonl') if keep(i['id'])]
    out.mkdir(parents=True)
    core.write_jsonl(out / 'texts.jsonl', texts)
    core.write_jsonl(out / 'lid.template.jsonl', template)
    core.write_jsonl(out / 'extract_issues.jsonl', issues)
    core.write_json(out / 'extract_manifest.json', dict(manifest, sample={
        'n': len(rids), 'seed': seed, 'rule': "random.Random(f'{seed}:{pair}').sample(range(1, total+1), n)",
        'records_with_items': len({r['record_id'] for r in texts}), 'items': len(texts),
        'issues': len(issues), 'full_texts_sha256': sha(full / 'texts.jsonl')}))
    core.write_jsonl(out / 'sampled_record_ids.jsonl', [{'record_id': r} for r in sorted(rids)])
    print(f'{pair}: {len(rids)} records sampled, {len({r["record_id"] for r in texts})} with items, '
          f'{len(texts)} items, {len(issues)} issues', flush=True)
    return texts


def to_offsets(lid):
    """LID row without text: spans are [start, end, lang] character offsets."""
    spans, pos = [], 0
    for s in lid['segments']:
        spans.append([pos, pos + len(s['text']), s['lang']])
        pos += len(s['text'])
    return {'sequence_id': lid['sequence_id'], 'text_sha256': lid['text_sha256'],
            'annotation_source': lid['annotation_source'], 'spans': spans}


def delivered_lid(core, path, texts):
    """Rebuild LID rows from offset labels and the CSV text; review() applies the review again."""
    rows = {r['sequence_id']: r for r in core.read_jsonl(path)}
    by_id = {t['sequence_id']: t for t in texts}
    check(set(rows) == set(by_id), f'{path} does not cover exactly the sampled sequences')
    lids = []
    for t in texts:
        row = rows[t['sequence_id']]
        check(row['text_sha256'] == t['text_sha256'], f'{t["sequence_id"]}: text checksum differs from {path}')
        check([a for a, _, _ in row['spans']] == [0] + [b for _, b, _ in row['spans'][:-1]]
              and row['spans'][-1][1] == len(t['text']), f'{t["sequence_id"]}: spans do not cover the text')
        lids.append({'sequence_id': row['sequence_id'], 'text_sha256': row['text_sha256'],
                     'annotation_source': row['annotation_source'], 'review_status': 'unreviewed',
                     'reviewer': '', 'segments': [{'text': t['text'][a:b], 'lang': lang}
                                                  for a, b, lang in row['spans']]})
    manifest_path = path.with_name('lid_manifest' + SUFFIX + '.json')
    original = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else None
    return lids, {'reused_from': str(path), 'reused_sha256': sha(path), 'original_manifest': original}


def review(core, pair, texts, lids, pilot):
    """Mark the items of the pilot records reviewed (no errors were found, so nothing changes)."""
    by_id = {t['sequence_id']: t for t in texts}
    result, rows = [], []
    for lid in lids:
        row = by_id[lid['sequence_id']]
        if row['record_id'] in pilot:
            rows.append([lid['sequence_id'], REVIEWER, REVIEW_DATE, 'reviewed', 'none', '', '', 'no change'])
            lid = dict(lid, review_status='reviewed', reviewer=REVIEWER)
        core.validate_lid(row, lid)
        result.append(lid)
    check({by_id[l['sequence_id']]['record_id'] for l in result if l['review_status'] == 'reviewed'} == pilot,
          f'{pair}: reviewed records differ from the pilot records')
    return result, rows


def cs_types(csv_path, texts):
    """sequence_id -> cs_type of the original record (evaluation metadata, never a model input)."""
    csv.field_size_limit(10**9)
    wanted, found = {r['source_row'] for r in texts}, {}
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        for number, row in enumerate(csv.DictReader(handle), 1):
            if number in wanted:
                found[number] = row['cs_type']
    return [{'sequence_id': r['sequence_id'], 'record_id': r['record_id'],
             'language_pair': r['language_pair'], 'cs_type': found[r['source_row']]} for r in texts]


def process(core, root, work, stage_dir, pair, args, assets):
    import annotate_script
    filename, code, _ = PAIRS[pair]
    raw = root / 'datasets' / filename
    base = work / pair
    full = base / 'extraction_full'
    if (full / 'extract_manifest.json').is_file():
        manifest = json.loads((full / 'extract_manifest.json').read_text(encoding='utf-8'))
        check(manifest['source_sha256'] == PAIRS[pair][2] and manifest['limit'] == 0,
              f'{full} does not match the full {filename}. Choose a new --work-dir.')
        print(f'{pair}: reuse full extraction in {full}', flush=True)
    else:
        core.extract(SimpleNamespace(input=str(raw), languages=f'{code},en', limit=0, out=str(full)))
    texts = sample(core, full, base / 'extraction', pair, args.seed, args.records)
    record_ids = sorted({t['record_id'] for t in texts})
    pilot = set(random.Random(f'{args.seed}:{pair}:pilot').sample(record_ids, min(args.pilot_records, len(record_ids))))

    delivered = relative(root, args.claude_lid_dir) / pair / 'annotation'
    if pair in CLAUDE_PAIRS:
        lids, lid_manifest = delivered_lid(core, delivered / f'lid_offsets{SUFFIX}.jsonl', texts)
        lid_source = 'delivered Claude labels (lid_offsets)'
    else:
        today = datetime.date.today().isoformat()
        lids, counts = annotate_script.annotate(texts, today)
        lid_manifest = {'method': 'unicode-script-rule', 'version': annotate_script.VERSION, 'date': today,
                        'sequences': len(lids), 'non_space_chars_by_label': dict(counts),
                        'script_sha256': sha(HERE / 'annotate_script.py')}
        lid_source = 'annotate_script.py'
    lids, review_rows = review(core, pair, texts, lids, pilot)

    annotation = base / 'annotation'
    annotation.mkdir(parents=True)
    core.write_jsonl(annotation / 'texts.jsonl', texts)
    core.write_jsonl(annotation / 'lid.jsonl', lids)
    core.tokenize(SimpleNamespace(texts=str(annotation / 'texts.jsonl'), lid=str(annotation / 'lid.jsonl'),
                                  tokenizer_dir=str(assets), neutral_policy='inherit_previous',
                                  end_policy='observed', allow_unreviewed=True, out=str(base / 'tokenized')))

    out = stage_dir / pair
    for sub in ['extraction', 'annotation', 'tokenized']:
        (out / sub).mkdir(parents=True)
    for name in ['texts.jsonl', 'lid.template.jsonl', 'extract_issues.jsonl', 'extract_manifest.json',
                 'sampled_record_ids.jsonl']:
        stem, ext = name.rsplit('.', 1)
        shutil.copy2(base / 'extraction' / name, out / 'extraction' / f'{stem}{SUFFIX}.{ext}')
    for name in ['labeled_tokens.jsonl', 'token_issues.jsonl', 'stats.json', 'manifest.json']:
        stem, ext = name.rsplit('.', 1)
        shutil.copy2(base / 'tokenized' / name, out / 'tokenized' / f'{stem}{SUFFIX}.{ext}')
    core.write_jsonl(out / 'extraction' / f'sequence_index{SUFFIX}.jsonl',
                     [{k: v for k, v in t.items() if k != 'text'} for t in texts])
    core.write_jsonl(out / 'annotation' / f'lid{SUFFIX}.jsonl', lids)
    core.write_json(out / 'annotation' / f'lid_manifest{SUFFIX}.json', lid_manifest)
    if pair in CLAUDE_PAIRS:
        core.write_jsonl(out / 'annotation' / f'lid_offsets{SUFFIX}.jsonl', [to_offsets(l) for l in lids])
        # The harmonization records belong to the Claude labeling run; they are text-free.
        for name in [f'harmonize_changes{SUFFIX}.jsonl', f'harmonize_manifest{SUFFIX}.json']:
            shutil.copy2(delivered / name, out / 'annotation' / name)
    (out / 'annotation' / f'pilot_record_ids{SUFFIX}.txt').write_text(
        ''.join(r + '\n' for r in sorted(pilot)), encoding='utf-8', newline='\n')
    with (out / 'annotation' / f'review{SUFFIX}.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sequence_id', 'reviewer', 'review_date', 'status', 'issue_type', 'before', 'after', 'reason'])
        writer.writerows(review_rows)
    core.write_json(out / 'annotation' / f'build_manifest{SUFFIX}.json', {
        'lid_source': lid_source, 'items': len(lids),
        'reviewed_items': sum(l['review_status'] == 'reviewed' for l in lids),
        'review_rows': len(review_rows), 'fixes': 0,
        'pilot_rule': "random.Random(f'{seed}:{pair}:pilot').sample(sorted record_ids with items, 20)",
        'runner_sha256': sha(Path(__file__))})
    core.write_jsonl(out / f'cs_type{SUFFIX}.jsonl', cs_types(raw, texts))
    stats = json.loads((base / 'tokenized/stats.json').read_text(encoding='utf-8'))
    print(f'{pair}: {stats["sequences"]} sequences, {stats["switches"]} switches', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root')
    parser.add_argument('--venv', default='project_update_1/.venv')
    parser.add_argument('--work-dir', default='project_update_1/data/yulun_reproduction',
                        help='Intermediate files; a finished full extraction here is reused')
    parser.add_argument('--out', default='deliverables/yulun_full', help='Full delivery with text; never commit it')
    parser.add_argument('--check-against', default='deliverables/yulun',
                        help="Committed text-free delivery to compare with; '' skips the check")
    parser.add_argument('--claude-lid-dir', default='deliverables/yulun',
                        help='Delivery whose <pair>/annotation/lid_offsets_yulun.jsonl holds the Claude labels')
    parser.add_argument('--pairs', nargs='+', choices=list(PAIRS), default=list(PAIRS))
    parser.add_argument('--records', type=int, default=500)
    parser.add_argument('--pilot-records', type=int, default=20)
    parser.add_argument('--seed', type=int, default=6140)
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--skip-install', action='store_true', help='Reuse the installed pinned dependencies')
    args = parser.parse_args()
    check(sys.version_info >= (3, 11), 'Python 3.11+ is required.')
    check(args.records > 0 and args.pilot_records > 0, 'Invalid --records or --pilot-records')
    root = repository(args.repo_root)
    standard = root / 'project_update_1'
    destination, work = relative(root, args.out), relative(root, args.work_dir)
    if not args.setup_only:
        check(not destination.exists(), f'{destination} already exists. Pass a new --out, for example deliverables/yulun_full2.')
        check(not destination.is_relative_to(work) and not work.is_relative_to(destination),
              '--out and --work-dir must be separate directories')
        for pair in args.pairs:
            filename, _, expected = PAIRS[pair]
            raw = root / 'datasets' / filename
            check(raw.is_file(), f'Missing datasets/{filename}. Download it from Hugging Face '
                                 'Shelton1013/SwitchLingua_text.')
            check(sha(raw) == expected, f'datasets/{filename} is not the expected SwitchLingua file (sha256)')
            if pair in CLAUDE_PAIRS:
                offsets = relative(root, args.claude_lid_dir) / pair / 'annotation' / f'lid_offsets{SUFFIX}.jsonl'
                check(offsets.is_file(), f'Missing {offsets}')

    env = relative(root, args.venv)
    executable = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if Path(sys.prefix).resolve() != env.resolve():
        if not executable.exists():
            check(not args.skip_install, '--skip-install requires an existing virtual environment')
            print(f'Create Python environment: {env}', flush=True)
            venv.EnvBuilder(with_pip=True).create(env)
        if not args.skip_install:
            subprocess.run([str(executable), '-m', 'pip', 'install', '-r',
                            str(standard / 'requirements.lock.txt')], check=True)
        command = [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]]
        if '--skip-install' not in command:
            command.append('--skip-install')
        sys.exit(subprocess.run(command).returncode)

    sys.path.insert(0, str(HERE))
    import pipeline as core
    assets = standard / 'assets/mbert'
    prepare_tokenizer(core, assets)
    if args.setup_only:
        print('SETUP COMPLETE: pinned dependencies and frozen tokenizer ready.', flush=True)
        return
    stage_dir = work / 'delivery_stage'
    earlier = [stage_dir] + [work / pair / sub for pair in args.pairs for sub in ['extraction', 'annotation', 'tokenized']]
    earlier = [str(p) for p in earlier if p.exists()]
    check(not earlier, f'Earlier run output exists: {earlier}. Delete it (extraction_full can stay) '
                       'or choose a new --work-dir.')
    for pair in args.pairs:
        process(core, root, work, stage_dir, pair, args, assets)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_dir.rename(destination)
    print(f'COMPLETE: {destination}. No Git operations.', flush=True)
    if args.check_against:
        compare(destination, relative(root, args.check_against), args.pairs)


def compare(destination, reference, pairs):
    """Compare the text-free committed files with the new run. Run-specific manifests are skipped.
    Line endings are ignored, because a Git checkout can change them."""
    lf = lambda path: path.read_bytes().replace(b'\r\n', b'\n')
    differ = []
    for pair in pairs:
        for ref in sorted((reference / pair).rglob('*')):
            name = ref.name
            if not ref.is_file() or ('manifest' in name and not name.startswith('extract_manifest')):
                continue
            new = destination / ref.relative_to(reference)
            if not new.is_file() or lf(new) != lf(ref):
                differ.append(str(ref.relative_to(reference)))
    if differ:
        raise ValueError(f'The new run differs from {reference} in: {differ}')
    print(f'CHECK PASS: text-free files match {reference}.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
