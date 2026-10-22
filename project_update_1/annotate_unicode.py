"""Rule-based LID by Unicode script, for pairs whose two languages use different scripts.

Each character gets a label from its script: the native script -> the native language code,
Latin letters -> en (Latin-script proper nouns count as en), digits/punctuation/symbols -> NEUTRAL,
any other script -> UNKNOWN. Whitespace takes the label of the character before it.
Runs of equal labels become lid.jsonl segments. No text is sent anywhere.

run_unicode_deepseek.py calls annotate(). Standalone use (from the repository root):
    python project_update_1/annotate_unicode.py --texts <texts.jsonl> --out <new directory>
"""
import argparse
import datetime
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import read_jsonl, write_jsonl, write_json, new_dir, validate_lid, digest  # noqa: E402

VERSION = 'v1.0'
# Han characters: CJK Unified Ideographs, Extension A-F, Compatibility Ideographs.
HAN = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)]
HANGUL = [(0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7AF), (0xD7B0, 0xD7FF)]
CYRILLIC = [(0x0400, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]
# Native scripts per language code. Hanja inside Korean text counts as Korean.
NATIVE = {'yue': HAN, 'zh': HAN, 'ko': HANGUL + HAN, 'ru': CYRILLIC}


def in_ranges(cp, ranges):
    return any(a <= cp <= b for a, b in ranges)


def char_label(ch, native_code):
    cp = ord(ch)
    if in_ranges(cp, NATIVE[native_code]):
        return native_code
    cat = unicodedata.category(ch)
    if cat.startswith('L'):
        # Latin letters, including accented and fullwidth forms.
        return 'en' if 'LATIN' in unicodedata.name(ch, '') else 'UNKNOWN'
    if cat.startswith(('N', 'P', 'S')) or cat in ('Mn', 'Me', 'Cf'):
        return 'NEUTRAL'
    return 'UNKNOWN'


def segments_for(text, native_code):
    labels, previous = [], 'NEUTRAL'
    for ch in text:
        lab = previous if ch.isspace() else char_label(ch, native_code)
        labels.append(lab)
        previous = lab
    segs = []
    for ch, lab in zip(text, labels):
        if segs and segs[-1]['lang'] == lab:
            segs[-1]['text'] += ch
        else:
            segs.append({'text': ch, 'lang': lab})
    return segs


def annotate(rows, date):
    """Return (lid rows, non-space character counts by label) for texts.jsonl rows."""
    lids, counts = [], Counter()
    for row in rows:
        native = next(c for c in row['languages'] if c != 'en')
        if native not in NATIVE or 'en' not in row['languages']:
            raise ValueError(f'Script rule needs en plus one of {sorted(NATIVE)}: {row["languages"]}')
        segs = segments_for(row['text'], native)
        lid = {'sequence_id': row['sequence_id'], 'text_sha256': row['text_sha256'],
               'annotation_source': f'script-rule/annotate_unicode.py-{VERSION}/{date}',
               'review_status': 'unreviewed', 'reviewer': '', 'segments': segs}
        validate_lid(row, lid)
        lids.append(lid)
        for s in segs:
            counts[s['lang']] += len(s['text'].strip())
    return lids, counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--texts', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    rows = read_jsonl(args.texts)
    date = datetime.date.today().isoformat()
    lids, counts = annotate(rows, date)
    out = new_dir(args.out)
    write_jsonl(out / 'texts.jsonl', rows)
    write_jsonl(out / 'lid.jsonl', lids)
    write_json(out / 'lid_manifest.json', {'method': 'unicode-script-rule', 'version': VERSION,
               'date': date, 'input_texts': args.texts, 'sequences': len(lids),
               'non_space_chars_by_label': dict(counts),
               'script_sha256': digest(Path(__file__).read_bytes())})
    print(f'Labeled {len(lids)} sequences; chars by label {dict(counts)}')


if __name__ == '__main__':
    main()
