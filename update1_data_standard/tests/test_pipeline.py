import argparse
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
import pipeline as p
from loader import iter_samples, batches


def row(sid='s1', rid='r1', text='aa bb', langs=None):
    langs = langs or ['fr', 'en']
    sw, dur, lengths, notes = p.labels(langs, 'observed')
    n = len(langs)
    return dict(sequence_id=sid, record_id=rid, language_pair='fr_en', text=text,
                text_sha256=p.digest(text.encode()), input_ids=list(range(100, 100+n)),
                tokens=['x']*n, offsets=[[0, 1]]*n, language_ids=langs,
                alignment_notes=['']*n, switch_labels=sw,
                switch_valid_mask=[v >= 0 for v in sw], duration_labels=dur,
                duration_valid_mask=[v >= 0 for v in dur], duration_lengths=lengths,
                label_notes=notes, end_policy='observed', neutral_policy='mask',
                tokenizer_sha256='fixture-not-a-real-tokenizer', review_status='unreviewed')


class Tests(unittest.TestCase):
    def test_bins_and_off_by_one(self):
        for length, target in [(1, 0), (2, 0), (3, 1), (6, 1), (7, 2), (9, 2)]:
            sw, dur, lengths, _ = p.labels(['fr']+['en']*length+['fr'], 'observed')
            self.assertEqual((sw[0], dur[0], lengths[0]), (1, target, length))
            self.assertEqual(sw[-1], -1)
            self.assertTrue(all(dur[i] == -1 for i in range(len(sw)) if sw[i] != 1))

    def test_unknown_and_censored(self):
        self.assertEqual(p.labels(['fr','en','UNKNOWN','fr'], 'observed')[1][0], -1)
        self.assertEqual(p.labels(['fr','en','en'], 'censored')[1][0], -1)
        self.assertEqual(p.labels(['fr','en','en'], 'observed')[2][0], 2)
        self.assertEqual(p.labels(['fr'], 'observed')[0], [-1])

    def test_lid_no_rewrite(self):
        tr = dict(text='Je work', text_sha256='abc', languages=['fr','en'])
        lid = dict(text_sha256='abc', annotation_source='test', review_status='unreviewed',
                   segments=[dict(text='Je ',lang='fr'), dict(text='work',lang='en')])
        self.assertEqual(p.validate_lid(tr,lid)[0], ['fr']*3+['en']*4)
        lid['segments'][0]['text'] = 'Je'
        with self.assertRaises(ValueError): p.validate_lid(tr,lid)

    def test_alignment_unicode_and_neutral(self):
        text = 'é!英'
        langs, _ = p.align(text,['fr','NEUTRAL','zh'],[(0,1),(1,2),(2,3)], [1,2,3],0,'inherit_previous')
        self.assertEqual(langs,['fr','fr','zh'])
        self.assertEqual(p.align(text,['fr','NEUTRAL','zh'],[(0,3)], [1],0,'mask')[0],['UNKNOWN'])
        self.assertEqual(p.align(text,['fr','NEUTRAL','zh'],[(0,1)], [0],0,'mask')[0],['UNKNOWN'])

    def test_extract_nested_and_malformed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with (root/'raw.csv').open('w',encoding='utf-8',newline='') as f:
                w = csv.writer(f); w.writerow(['data_generation_result'])
                w.writerow([repr(['Bonjour\nhello'])])
                w.writerow([repr([{'topic':'x','instances':['é English']}])])
                w.writerow(['not a literal'])
            p.extract(argparse.Namespace(input=root/'raw.csv', languages='fr,en',limit=0,out=root/'out'))
            rows = p.read_jsonl(root/'out/texts.jsonl')
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['text'],'Bonjour\nhello')
            self.assertEqual(rows[1]['source_path'],[0,'instances',0])
            self.assertEqual(len(p.read_jsonl(root/'out/extract_issues.jsonl')),1)

    def test_group_split_and_loader_no_future(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [row(f's{i}', f'r{i}', f'text {i}', ['fr','en','en']) for i in range(20)]
            rows.append(row('extra','r0','another text',['fr','en','en']))
            rows.append(row('duplicate','r100','text 0',['fr','en','en']))
            p.write_jsonl(root/'data.jsonl',rows)
            p.split(argparse.Namespace(inputs=[root/'data.jsonl'],seed=6140,out=root/'split'))
            splits = p.unique(p.read_jsonl(root/'split/splits.jsonl'),'sequence_id')
            self.assertEqual(splits['s0']['split'],splits['extra']['split'])
            self.assertEqual(splits['s0']['group_id'],splits['duplicate']['group_id'])
            for mode in ['current','prefix']:
                items = list(iter_samples(root/'split/labeled_tokens.jsonl',root/'split/splits.jsonl','train',mode))
                self.assertTrue(items)
                for sample in items:
                    t=sample['provenance']['position']
                    ids=sample['inputs']['input_ids']
                    self.assertEqual(ids[-1],100+t)
                    self.assertEqual(len(ids),1 if mode=='current' else t+1)
                    self.assertNotIn(100+t+1,ids)
                    self.assertEqual(set(sample['inputs']),{'input_ids'})
            self.assertEqual(len(list(batches(range(5),2))),3)

    def test_validation_catches_bad_labels(self):
        r=row();p.validate_row(r)
        r['switch_labels'][0]=0
        with self.assertRaises(ValueError):p.validate_row(r)

    def test_pilot_group_forced_train(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [row(f's{i}',f'r{i}',f'hello {i}') for i in range(10)]
            rows.append(row('copy','copy-record','hello 0'))
            p.write_jsonl(root/'data.jsonl',rows)
            (root/'pilot.txt').write_text('r0\n')
            p.split(argparse.Namespace(inputs=[root/'data.jsonl'],seed=6140,
                                      out=root/'split',train_records=root/'pilot.txt'))
            entries=p.unique(p.read_jsonl(root/'split/splits.jsonl'),'sequence_id')
            self.assertEqual(entries['s0']['split'],'train')
            self.assertEqual(entries['copy']['split'],'train')


if __name__ == '__main__':
    unittest.main()
