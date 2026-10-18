"""Build the unreviewed personal delivery; performs no Git mutations."""
import ast
import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from pipeline import validate_lid, validate_row, summary

SOURCE = Path('deliverables/yufan')
OUTPUT = Path('deliverables/.yufan_submission')
ARCHIVE = Path('data/archive/2026-10-17/pre_submission_layout_yufan')
RAW = Path('../SwitchLingua')
PAIRS = {'fr_en': 'French_eng.csv', 'de_en': 'German_Eng.csv',
         'it_en': 'Italian_eng.csv', 'philippines_en': 'Philippines_eng.csv'}


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def jsonl(path, records):
    with path.open('x', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    global SOURCE, OUTPUT, RAW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(SOURCE))
    parser.add_argument('--out', default=str(OUTPUT))
    parser.add_argument('--raw-dir', default=str(RAW))
    args = parser.parse_args()
    SOURCE, OUTPUT, RAW = Path(args.source), Path(args.out), Path(args.raw_dir)
    assert SOURCE.is_dir() and not OUTPUT.exists()
    git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    original_hashes = {str(p): sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    OUTPUT.mkdir()
    scripts = OUTPUT / 'scripts'
    scripts.mkdir()
    shutil.copy2('annotate_lingua.py', scripts / 'annotate_lingua.py')
    shutil.copy2(__file__, scripts / 'prepare_submission.py')
    runtime_names = [line.split('==')[0] for line in Path('requirements.lock.txt').read_text().splitlines()
                     if '==' in line and not line.startswith('#')] + ['lingua-language-detector']
    versions = {name: importlib.metadata.version(name) for name in runtime_names}
    (OUTPUT / 'requirements_yufan.txt').write_text(''.join(f'{name}=={value}\n' for name, value in versions.items()))
    save(OUTPUT / 'runtime_yufan.json', {
        'shared_code_commit': git_head, 'branch': 'data/yufan',
        'shared_code_sha256': {name: sha(Path(name)) for name in ['pipeline.py', 'loader.py', 'LID_PROMPT.md']},
        'python': __import__('platform').python_version(), 'versions': versions,
        'tokenizer': json.loads(Path('assets/mbert/tokenizer_manifest.json').read_text()),
        'neutral_policy': 'inherit_previous', 'end_policy': 'observed',
        'sequence_policy': 'one extracted item per sequence', 'independent_linguistic_verification': False,
    })
    known_development = {'fr_en': set(), 'de_en': set(), 'it_en': set(), 'philippines_en': set()}
    fr_reference = Path('data/archive/2026-10-17/pre_quarantine_delivery_yufan/fr_en/after_annotation/texts_yufan.jsonl')
    known_development['fr_en'].update(r['record_id'] for r in read(fr_reference))
    audit = Path('data/lid_review/de_it_lingua_2026-10-17/inspection.json')
    for pair, data in json.loads(audit.read_text()).items():
        for row in data['random_samples'] + data['possible_missed_short_english_examples']:
            known_development[pair].add(row['id'].rsplit(':', 1)[0])
    totals = {}
    for pair, filename in PAIRS.items():
        current = SOURCE / pair
        kept = current / 'after_annotation/retained'
        text_rows = read(kept / 'texts_yufan.jsonl')
        lids = read(kept / 'lid_yufan.jsonl')
        tokens = read(kept / 'labeled_tokens_yufan.jsonl')
        assert len(text_rows) == len(lids) == len(tokens)
        assert len({r['sequence_id'] for r in tokens}) == len(tokens)
        for text, lid, token in zip(text_rows, lids, tokens):
            assert text['sequence_id'] == lid['sequence_id'] == token['sequence_id']
            assert text['text'] == token['text'] and lid['review_status'] == token['review_status'] == 'unreviewed'
            validate_lid(text, lid)
            validate_row(token)
        raw_path = RAW / filename
        raw_hash = sha(raw_path)
        with raw_path.open(encoding='utf-8-sig', newline='') as handle:
            original_rows = list(csv.DictReader(handle))
        parsed = {}
        cs_rows = []
        for text in text_rows:
            row_number = text['source_row']
            original = original_rows[row_number - 1]
            assert text['source_file'] == filename
            assert text['record_id'] == f'{pair}:{raw_hash[:12]}:{row_number:06d}'
            assert original['cs_type'] in {'Intersentential', 'Intrasentential', 'Extra-sentential / Tag switching'}
            if row_number not in parsed:
                cell = original['data_generation_result']
                try:
                    parsed[row_number] = json.loads(cell)
                except json.JSONDecodeError:
                    parsed[row_number] = ast.literal_eval(cell)
            value = parsed[row_number]
            for step in text['source_path']:
                value = value[step]
            assert value == text['text'], f'Raw text mismatch: {text["sequence_id"]}'
            cs_rows.append({'sequence_id': text['sequence_id'], 'record_id': text['record_id'],
                            'source_row': row_number, 'cs_type': original['cs_type']})
        folder = OUTPUT / pair
        for name in ['extraction', 'annotation', 'tokenized']:
            (folder / name).mkdir(parents=True)
        shutil.copy2(kept / 'texts_yufan.jsonl', folder / 'extraction/texts_yufan.jsonl')
        templates = {r['sequence_id']: r for r in read(current / 'before_annotation/lid.template_yufan.jsonl')}
        jsonl(folder / 'extraction/lid.template_yufan.jsonl', [templates[r['sequence_id']] for r in text_rows])
        shutil.copy2(current / 'before_annotation/extract_issues_yufan.jsonl', folder / 'extraction/extract_issues_yufan.jsonl')
        full_manifest = json.loads((current / 'before_annotation/extract_manifest_yufan.json').read_text())
        save(folder / 'extraction/extract_manifest_yufan.json', {
            'operation': 'select final retained sequences from existing full extraction',
            'source_file': filename, 'source_sha256': raw_hash, 'full_extraction': full_manifest,
            'delivered_items': len(tokens), 'all_delivered_texts_verified_against_raw_csv': True,
            'text_modified': False, 'selection': 'See annotation/screening_manifest_yufan.json',
            'extraction_issues_scope': 'Full source extraction, not only delivered sequences',
        })
        shutil.copy2(kept / 'lid_yufan.jsonl', folder / 'annotation/lid_yufan.jsonl')
        for name in ['labeled_tokens_yufan.jsonl', 'token_issues_yufan.jsonl', 'stats_yufan.json']:
            shutil.copy2(kept / name, folder / 'tokenized' / name)
        assert summary(tokens) == json.loads((folder / 'tokenized/stats_yufan.json').read_text())
        jsonl(folder / 'cs_type_yufan.jsonl', cs_rows)
        full_lid_manifest = json.loads((Path('data') / pair / 'lingua_lid/lid_manifest.json').read_text())
        save(folder / 'annotation/lid_manifest_yufan.json', {
            'operation': 'select existing unreviewed Lingua preannotations; no relabeling',
            'source_annotation_run': full_lid_manifest, 'delivered_sequences': len(tokens),
            'lid_sha256': sha(folder / 'annotation/lid_yufan.jsonl'),
            'independently_verified': False,
        })
        screen_path = current / 'after_annotation/screening_summary_yufan.json'
        screen_summary = json.loads(screen_path.read_text())
        screen_rules = (json.loads((current / 'after_annotation/screening_rules_yufan.json').read_text())
                        if pair == 'philippines_en' else json.loads((SOURCE / 'screening_rules_yufan.json').read_text()))
        save(folder / 'annotation/screening_manifest_yufan.json', {
            'summary': screen_summary, 'rules': screen_rules,
            'retained_status': 'not flagged by these rules, NOT independently verified',
            'prior_full_and_quarantined_data': 'Kept locally; not part of this submission payload',
        })
        q = current / 'after_annotation/quarantined'
        shutil.copy2(q / 'reasons_yufan.jsonl', folder / 'annotation/quarantine_reasons_yufan.jsonl')
        with (q / 'review_yufan.csv').open(encoding='utf-8-sig', newline='') as src, \
                (folder / 'annotation/review_yufan.csv').open('w', encoding='utf-8', newline='') as dst:
            writer = csv.writer(dst)
            writer.writerow(['sequence_id', 'reviewer', 'review_date', 'status', 'issue_type', 'before', 'after', 'reason'])
            for row in csv.DictReader(src):
                writer.writerow([row['sequence_id'], '', '', 'needs_review', row['reasons'], row['text'], '',
                                 'Automated quarantine only; no human linguistic adjudication.'])
        if pair == 'philippines_en':
            example_text = 'Nakaka-excite ang dynamic weather effects at anime-style visuals na nagpapaganda ng overall experience habang nilalaro.'
            for row in read(q / 'texts_yufan.jsonl'):
                if row['text'] == example_text:
                    known_development[pair].add(row['record_id'])
        record_ids = {r['record_id'] for r in text_rows}
        present_dev = sorted(record_ids & known_development[pair])
        absent_dev = sorted(known_development[pair] - record_ids)
        (folder / 'annotation/pilot_record_ids_yufan.txt').write_text(''.join(r + '\n' for r in present_dev))
        save(folder / 'annotation/development_record_manifest_yufan.json', {
            'meaning': 'Records already used in development examples or model-assisted comparisons; NOT a reviewed gold set',
            'present_in_delivery': present_dev, 'absent_from_delivery': absent_dev,
        })
        token_run = json.loads((Path('data') / pair / 'lingua_tokenized/manifest.json').read_text())
        save(folder / 'tokenized/manifest_yufan.json', {
            'operation': 'select existing tokenized rows; no retokenization or label modification',
            'source_tokenization_run': token_run, 'selection': screen_summary,
            'delivered_sequences': len(tokens), 'labeled_tokens_sha256': sha(folder / 'tokenized/labeled_tokens_yufan.jsonl'),
            'texts_sha256': sha(folder / 'extraction/texts_yufan.jsonl'),
            'lid_sha256': sha(folder / 'annotation/lid_yufan.jsonl'),
            'cs_type_sha256': sha(folder / 'cs_type_yufan.jsonl'),
            'tokenizer': json.loads(Path('assets/mbert/tokenizer_manifest.json').read_text()),
            'neutral_policy': 'inherit_previous', 'end_policy': 'observed',
            'review_status': 'unreviewed', 'independent_linguistic_verification': False,
        })
        candidates = read(q / 'reasons_yufan.jsonl')
        case = next((r for r in candidates if 'english_labeled_at_requires_context_review' in r['reasons']), candidates[0])
        case_text = next(r for r in read(q / 'texts_yufan.jsonl') if r['sequence_id'] == case['sequence_id'])
        save(folder / 'annotation/case_study_yufan.json', {
            'original_text': case_text, 'screening_result': case,
            'action': 'Quarantined whole sequence; no language label corrected',
            'remaining_uncertainty': 'Flag is an automatic review candidate, not an adjudicated linguistic error',
        })
        totals[pair] = {'sequences': len(tokens), 'records': len(record_ids), 'stats': summary(tokens),
                        'cs_type_counts': dict(Counter(r['cs_type'] for r in cs_rows)),
                        'quarantined_sequences': screen_summary['quarantined_total'],
                        'development_records_in_delivery': len(present_dev)}
        print(pair, len(tokens), 'PASS: raw text, ID/LID/token alignment, cs_type, stats', flush=True)
        del tokens, text_rows, lids
    save(OUTPUT / 'summary_yufan.json', totals)
    # Source files have not been modified. Only the layout entry point changes.
    assert all(sha(Path(path)) == value for path, value in original_hashes.items())
    print('BUILD COMPLETE. Original retained/quarantined layout not moved yet.', flush=True)


if __name__ == '__main__':
    main()
