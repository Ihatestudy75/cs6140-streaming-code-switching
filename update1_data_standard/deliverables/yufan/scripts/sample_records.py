"""Sample whole records from a prepared delivery. No Git operations."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from pipeline import validate_lid, validate_row, summary

PAIRS = ['fr_en', 'de_en', 'it_en', 'philippines_en']


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(s) for s in handle if s.strip()]


def write(path, records):
    with path.open('x', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--records', type=int, default=500)
    parser.add_argument('--seed', type=int, default=6140)
    args = parser.parse_args()
    source, out = Path(args.source), Path(args.out)
    assert args.records > 0 and source.is_dir() and not out.exists()
    source_hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    out.mkdir()
    for p in source.iterdir():
        if p.is_file() and p.name not in ['summary_yufan.json', 'file_inventory_yufan.json']:
            shutil.copy2(p, out / p.name)
    shutil.copytree(source / 'scripts', out / 'scripts')
    shutil.copy2(__file__, out / 'scripts/sample_records.py')
    totals = {}
    for pair in PAIRS:
        src = source / pair
        texts = read(src / 'extraction/texts_yufan.jsonl')
        record_ids = sorted({r['record_id'] for r in texts})
        assert len(record_ids) >= args.records
        selected = set(random.Random(f'{args.seed}:{pair}').sample(record_ids, args.records))
        wanted_texts = [r for r in texts if r['record_id'] in selected]
        ids = {r['sequence_id'] for r in wanted_texts}
        folder = out / pair
        for sub in ['extraction', 'annotation', 'tokenized']:
            (folder / sub).mkdir(parents=True)
            for p in (src / sub).iterdir():
                if p.suffix != '.jsonl':
                    shutil.copy2(p, folder / sub / p.name)
        # Preserve source row order and exact JSON objects for all selected IDs.
        for sub, name in [('extraction', 'texts_yufan.jsonl'), ('extraction', 'lid.template_yufan.jsonl'),
                          ('annotation', 'lid_yufan.jsonl'), ('tokenized', 'labeled_tokens_yufan.jsonl'),
                          ('tokenized', 'token_issues_yufan.jsonl')]:
            records = [r for r in read(src / sub / name) if r['sequence_id'] in ids]
            write(folder / sub / name, records)
        shutil.copy2(src / 'extraction/extract_issues_yufan.jsonl', folder / 'extraction/extract_issues_yufan.jsonl')
        write(folder / 'cs_type_yufan.jsonl', [r for r in read(src / 'cs_type_yufan.jsonl') if r['sequence_id'] in ids])
        write(folder / 'annotation/quarantine_reasons_yufan.jsonl',
              [r for r in read(src / 'annotation/quarantine_reasons_yufan.jsonl') if r['record_id'] in selected])
        with (src / 'annotation/review_yufan.csv').open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            review_rows = [r for r in reader if r['sequence_id'].rsplit(':', 1)[0] in selected]
            fields = reader.fieldnames
        with (folder / 'annotation/review_yufan.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(review_rows)
        pilots = set((src / 'annotation/pilot_record_ids_yufan.txt').read_text().splitlines())
        (folder / 'annotation/pilot_record_ids_yufan.txt').write_text(''.join(r + '\n' for r in sorted(pilots & selected)))
        dev = json.loads((src / 'annotation/development_record_manifest_yufan.json').read_text())
        dev['not_selected_in_this_sample'] = sorted(set(dev['present_in_delivery']) - selected)
        dev['present_in_delivery'] = sorted(pilots & selected)
        save(folder / 'annotation/development_record_manifest_yufan.json', dev)
        sampled_tokens = read(folder / 'tokenized/labeled_tokens_yufan.jsonl')
        lids = read(folder / 'annotation/lid_yufan.jsonl')
        for text, lid, token in zip(wanted_texts, lids, sampled_tokens):
            assert text['sequence_id'] == lid['sequence_id'] == token['sequence_id']
            validate_lid(text, lid)
            validate_row(token)
        assert len(wanted_texts) == len(lids) == len(sampled_tokens) == len(ids)
        sample_manifest = {
            'method': 'random sample without replacement from sorted original record_ids in current retained delivery',
            'seed_expression': f'{args.seed}:{pair}', 'eligible_records': len(record_ids),
            'sampled_records': args.records, 'source_retained_sequences': len(texts),
            'sampled_sequences': len(ids), 'selected_record_ids': sorted(selected),
            'all_existing_retained_items_of_selected_records_included': True,
            'sampling_population': 'screened retained set, NOT the entire raw CSV',
            'labels_modified': False, 'review_status': 'unreviewed',
            'source_texts_sha256': sha(src / 'extraction/texts_yufan.jsonl'),
        }
        save(folder / 'extraction/sampling_manifest_yufan.json', sample_manifest)
        for sub, name in [('extraction', 'extract_manifest_yufan.json'), ('annotation', 'lid_manifest_yufan.json'),
                          ('tokenized', 'manifest_yufan.json')]:
            original = json.loads((src / sub / name).read_text())
            save(folder / sub / name, {'operation': 'sample existing records without editing text or labels',
                                      'source_manifest': original, 'sampled_records': args.records,
                                      'delivered_sequences': len(ids),
                                      'sampling_manifest': '../extraction/sampling_manifest_yufan.json'})
        token_manifest = json.loads((folder / 'tokenized/manifest_yufan.json').read_text())
        token_manifest['output_sha256'] = {name: sha(folder / sub / name) for sub, name in [
            ('extraction', 'texts_yufan.jsonl'), ('annotation', 'lid_yufan.jsonl'),
            ('tokenized', 'labeled_tokens_yufan.jsonl'), ('', 'cs_type_yufan.jsonl')]}
        save(folder / 'tokenized/manifest_yufan.json', token_manifest)
        stats = summary(sampled_tokens)
        save(folder / 'tokenized/stats_yufan.json', stats)
        counts = __import__('collections').Counter(r['cs_type'] for r in read(folder / 'cs_type_yufan.jsonl'))
        totals[pair] = {'sequences': len(ids), 'records': args.records, 'stats': stats,
                        'cs_type_counts': dict(counts), 'development_records_in_delivery': len(pilots & selected),
                        'source_retained_sequences': len(texts), 'source_eligible_records': len(record_ids)}
        assert len({r['record_id'] for r in wanted_texts}) == args.records
        assert ids == {r['sequence_id'] for r in texts if r['record_id'] in selected}
        print(pair, args.records, 'records,', len(ids), 'sequences', flush=True)
    save(out / 'summary_yufan.json', totals)
    assert all(sha(source / name) == value for name, value in source_hashes.items())
    print('PASS: complete selected records; original source unchanged; no Git operations.')


if __name__ == '__main__':
    main()
