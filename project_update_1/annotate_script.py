"""Rule-based LID by Unicode script for ar_en, zh_en, ja_en and the Devanagari part of hi_en.

Each character gets a label from its script: the native script -> the native language code,
Latin letters -> en (Latin-script proper nouns count as en), digits/punctuation/symbols -> NEUTRAL,
any other script -> UNKNOWN. Whitespace takes the label of the character before it.
Runs of equal labels become lid.jsonl segments. No text is sent anywhere.

The rule and char_label()/segments_for() are the same as annotate_unicode.py; this file adds the
Arabic, Devanagari and kana ranges and keeps their script-specific punctuation and digits NEUTRAL
(see deliverables/yulun/DECISIONS.md, section 3.2).

run_unicode_claude.py calls annotate(); annotate_claude.py --script-native calls segments_for().
Standalone use (from the repository root):
    python project_update_1/annotate_script.py --texts <texts.jsonl> --out <new directory>
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
# The ranges below cover letters and marks only. Script-specific punctuation and digits that sit
# inside these Unicode blocks are left out on purpose, so char_label() falls through to their
# general category and labels them NEUTRAL, the same as ASCII punctuation and digits. Otherwise
# e.g. `you know؟` would get a one-token en->ar switch at the question mark.
# Arabic block minus ، ؛ ؟ ۔, Arabic-Indic digits and separators (0660-066D), and the extended
# Arabic-Indic digits (06F0-06F9); plus Supplement, Extended-A and Presentation Forms A/B.
ARABIC = [(0x0600, 0x060B), (0x060D, 0x061A), (0x061C, 0x061E), (0x0620, 0x065F), (0x066E, 0x06D3),
          (0x06D5, 0x06EF), (0x06FA, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF),
          (0xFE70, 0xFEFF)]
# Devanagari block minus the danda । ॥, Devanagari digits and the abbreviation sign (0964-0970);
# plus Devanagari Extended.
DEVANAGARI = [(0x0900, 0x0963), (0x0971, 0x097F), (0xA8E0, 0xA8FF)]
# Hiragana, Katakana (minus the middle dot ・ U+30FB), Katakana Phonetic Extensions, halfwidth
# katakana including the halfwidth voiced marks, and the iteration/closing marks 々 〆, which
# would otherwise be UNKNOWN letters. Katakana loanwords such as コンピューター are labeled ja:
# correct by script even though they are English borrowings. Known limitation, not a bug.
KANA = [(0x3005, 0x3006), (0x3040, 0x309F), (0x30A0, 0x30FA), (0x30FC, 0x30FF), (0x31F0, 0x31FF),
        (0xFF66, 0xFF9F)]
# Native scripts per language code. Kanji inside Japanese text counts as Japanese.
NATIVE = {'zh': HAN, 'ar': ARABIC, 'hi': DEVANAGARI, 'ja': KANA + HAN}


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
               'annotation_source': f'script-rule/annotate_script.py-{VERSION}/{date}',
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
