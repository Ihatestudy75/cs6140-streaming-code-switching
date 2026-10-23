"""Tests for annotate_script.py on the different-script pairs ar/hi/zh/ja + en.

Run from project_update_1/:  python -m unittest discover -s tests -v
Every case checks that the segments reproduce the input character for character and pass the
shared validate_lid() unchanged.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import annotate_script  # noqa: E402
from annotate_script import char_label, segments_for  # noqa: E402
from pipeline import digest, read_jsonl, validate_lid, write_jsonl  # noqa: E402


def textrow(text, native):
    return {'sequence_id': 'test:000000000000:000001:001', 'record_id': 'test:000000000000:000001',
            'languages': [native, 'en'], 'text': text, 'text_sha256': digest(text.encode('utf-8'))}


class SegmentTests(unittest.TestCase):
    def check(self, text, native, expected):
        segs = segments_for(text, native)
        self.assertEqual(''.join(s['text'] for s in segs), text)
        self.assertEqual([(s['text'], s['lang']) for s in segs], expected)
        validate_lid(textrow(text, native), {'text_sha256': digest(text.encode('utf-8')),
                                             'annotation_source': 'test', 'review_status': 'unreviewed',
                                             'segments': segs})

    def test_arabic_english_one_switch(self):
        self.check('أنا أحب this game', 'ar', [('أنا أحب ', 'ar'), ('this game', 'en')])

    def test_hindi_english_one_switch(self):
        # Vowel signs and virama (Mn/Mc) are inside the Devanagari range, so words stay whole.
        self.check('मुझे लगता है that it works', 'hi', [('मुझे लगता है ', 'hi'), ('that it works', 'en')])

    def test_chinese_english_multiple_switches(self):
        self.check('我觉得这个term真的很confusing啊', 'zh',
                   [('我觉得这个', 'zh'), ('term', 'en'), ('真的很', 'zh'), ('confusing', 'en'), ('啊', 'zh')])

    def test_japanese_hiragana_katakana_han(self):
        # コンピューター is an English loanword but is labeled ja by script (documented limitation).
        text = '昨日コンピューターを買いました。So happy!'
        self.check(text, 'ja', [('昨日コンピューターを買いました', 'ja'), ('。', 'NEUTRAL'),
                                ('So happy', 'en'), ('!', 'NEUTRAL')])
        for ch in '昨コーを':  # han, katakana, prolonged sound mark, hiragana
            self.assertEqual(char_label(ch, 'ja'), 'ja')

    def test_punctuation_at_switch_boundary(self):
        # Pattern seen in the Arabic pilot: Arabic comma right before the switch.
        self.check('الفيلم، honestly', 'ar', [('الفيلم', 'ar'), ('، ', 'NEUTRAL'), ('honestly', 'en')])
        self.check('他说，OK', 'zh', [('他说', 'zh'), ('，', 'NEUTRAL'), ('OK', 'en')])

    def test_native_punctuation_after_english_is_neutral(self):
        # The case the range split exists for: no ar/hi label on the final mark, so no false switch.
        self.check('you know؟', 'ar', [('you know', 'en'), ('؟', 'NEUTRAL')])
        self.check('यह अच्छा है। Really good.', 'hi',
                   [('यह अच्छा है', 'hi'), ('। ', 'NEUTRAL'), ('Really good', 'en'), ('.', 'NEUTRAL')])

    def test_text_ending_in_punctuation(self):
        self.check('真的很好 right?', 'zh', [('真的很好 ', 'zh'), ('right', 'en'), ('?', 'NEUTRAL')])

    def test_digits_and_emoji(self):
        # ❤️ is U+2764 plus variation selector U+FE0F (Mn); both are NEUTRAL.
        self.check('I scored 100分 today 😀❤️', 'zh',
                   [('I scored ', 'en'), ('100', 'NEUTRAL'), ('分 ', 'zh'), ('today ', 'en'),
                    ('😀❤️', 'NEUTRAL')])

    def test_embedded_newline(self):
        self.check('Hello\nمرحبا', 'ar', [('Hello\n', 'en'), ('مرحبا', 'ar')])
        self.check('\nHello', 'ar', [('\n', 'NEUTRAL'), ('Hello', 'en')])

    def test_latin_proper_noun_inside_native_text(self):
        # Same convention as annotate_unicode.py: Latin-script names are labeled en.
        self.check('شاهدت فيلم Marvel الجديد', 'ar',
                   [('شاهدت فيلم ', 'ar'), ('Marvel ', 'en'), ('الجديد', 'ar')])
        self.check('我觉得这支球队表现不错，especially 小王！', 'zh',
                   [('我觉得这支球队表现不错', 'zh'), ('，', 'NEUTRAL'), ('especially ', 'en'),
                    ('小王', 'zh'), ('！', 'NEUTRAL')])

    def test_arabic_indic_and_devanagari_digits_are_neutral(self):
        # These sit inside the Arabic/Devanagari blocks but are excluded from the ranges.
        self.check('عندي ٣ كتب', 'ar', [('عندي ', 'ar'), ('٣ ', 'NEUTRAL'), ('كتب', 'ar')])
        self.check('عندي ۳ كتب', 'ar', [('عندي ', 'ar'), ('۳ ', 'NEUTRAL'), ('كتب', 'ar')])
        self.check('कुल ३ लोग', 'hi', [('कुल ', 'hi'), ('३ ', 'NEUTRAL'), ('लोग', 'hi')])

    def test_japanese_iteration_mark_and_middle_dot(self):
        self.check('人々はhappyです', 'ja', [('人々は', 'ja'), ('happy', 'en'), ('です', 'ja')])
        self.check('ジョン・スミス', 'ja', [('ジョン', 'ja'), ('・', 'NEUTRAL'), ('スミス', 'ja')])

    def test_other_script_is_unknown(self):
        # Telugu letters appear in a few Hindi rows; they are not Devanagari. The Telugu vowel sign
        # is a combining mark (Mn) and so NEUTRAL, but align() still makes the subword UNKNOWN
        # because it contains an UNKNOWN character.
        self.check('నా friend', 'hi', [('న', 'UNKNOWN'), ('ా ', 'NEUTRAL'), ('friend', 'en')])


class MainTests(unittest.TestCase):
    def run_main(self, rows, tmp):
        texts = Path(tmp) / 'texts.jsonl'
        write_jsonl(texts, rows)
        out = Path(tmp) / 'out'
        with mock.patch.object(sys, 'argv', ['annotate_script.py', '--texts', str(texts), '--out', str(out)]), \
                mock.patch('builtins.print'):
            annotate_script.main()
        return out

    def test_end_to_end_copies_ids_and_hashes(self):
        rows = [dict(textrow('أعتقد أن الجميع لديهم رأي، you know?', 'ar'), sequence_id='a:1'),
                dict(textrow('This whole story sounds like a soap opera.', 'ar'), sequence_id='a:2')]
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_main(rows, tmp)
            lids = read_jsonl(out / 'lid.jsonl')
            self.assertEqual(read_jsonl(out / 'texts.jsonl'), rows)
            manifest = json.loads((out / 'lid_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['sequences'], 2)
        for row, lid in zip(rows, lids):
            self.assertEqual((lid['sequence_id'], lid['text_sha256']), (row['sequence_id'], row['text_sha256']))
            self.assertEqual(lid['review_status'], 'unreviewed')
            self.assertTrue(lid['annotation_source'].startswith('script-rule/annotate_script.py-'))
            validate_lid(row, lid)

    def test_same_script_pairs_are_refused(self):
        # Phase B pairs: both languages are Latin, so the script rule must not produce labels.
        for native in ('es', 'vi'):
            with self.subTest(native=native), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, 'Script rule needs en'):
                    self.run_main([textrow('Hola, how are you?', native)], tmp)
                self.assertFalse((Path(tmp) / 'out').exists())


if __name__ == '__main__':
    unittest.main()
