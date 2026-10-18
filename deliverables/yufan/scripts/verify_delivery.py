"""Read-only verification of the prepared delivery. No Git operations."""
import hashlib
import json
import sys
from pathlib import Path

DELIVERY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DELIVERY.parents[1]))
from pipeline import read_jsonl, validate_lid, validate_row, summary


def main():
    inventory = json.loads((DELIVERY / 'file_inventory_yufan.json').read_text())
    for item in inventory['files']:
        path = DELIVERY / item['path']
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == item['sha256'], path
        assert path.stat().st_size == item['bytes']
    expected = json.loads((DELIVERY / 'summary_yufan.json').read_text())
    counts = {}
    for pair in ['fr_en', 'de_en', 'it_en', 'philippines_en']:
        folder = DELIVERY / pair
        texts = read_jsonl(folder / 'extraction/texts_yufan.jsonl')
        lids = read_jsonl(folder / 'annotation/lid_yufan.jsonl')
        tokens = read_jsonl(folder / 'tokenized/labeled_tokens_yufan.jsonl')
        types = read_jsonl(folder / 'cs_type_yufan.jsonl')
        templates = read_jsonl(folder / 'extraction/lid.template_yufan.jsonl')
        assert len(texts) == len(lids) == len(tokens) == len(types) == len(templates)
        assert len({r['sequence_id'] for r in texts}) == len(texts)
        for text, lid, token, cs_type, template in zip(texts, lids, tokens, types, templates):
            assert text['sequence_id'] == lid['sequence_id'] == token['sequence_id'] == cs_type['sequence_id'] == template['sequence_id']
            assert text['text'] == token['text']
            assert text['record_id'] == token['record_id'] == cs_type['record_id']
            assert text['text_sha256'] == template['text_sha256']
            assert lid['review_status'] == token['review_status'] == 'unreviewed'
            validate_lid(text, lid)
            validate_row(token)
        assert summary(tokens) == json.loads((folder / 'tokenized/stats_yufan.json').read_text())
        records = {r['record_id'] for r in texts}
        sample = json.loads((folder / 'extraction/sampling_manifest_yufan.json').read_text())
        assert len(records) == sample['sampled_records'] == expected[pair]['records'] == 500
        assert records == set(sample['selected_record_ids'])
        assert len(tokens) == sample['sampled_sequences'] == expected[pair]['sequences']
        pilots = (folder / 'annotation/pilot_record_ids_yufan.txt').read_text().splitlines()
        assert set(pilots) <= records
        assert all(r['cs_type'] in {'Intersentential', 'Intrasentential', 'Extra-sentential / Tag switching'} for r in types)
        counts[pair] = len(tokens)
        print(f'PASS: {pair}: {len(tokens)} unreviewed sequences; cs_type/IDs/text/LID/labels/stats/checksums')
    total = sum(r['sequences'] for r in expected.values())
    assert sum(counts.values()) == total
    print(f'PASS: {total} sequences, 500 records per pair. Structure/provenance only; NOT linguistic accuracy.')


if __name__ == '__main__':
    main()
