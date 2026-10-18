"""Offline, unreviewed LID preannotations compatible with pipeline.py.

This generates language spans only, not subwords or prediction labels.
Lingua boundaries are heuristic and have not been linguistically validated.
"""
import argparse
import hashlib
import importlib.metadata
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from lingua import Language, LanguageDetectorBuilder
from pipeline import validate_lid


LANGUAGES = {'fr': Language.FRENCH, 'de': Language.GERMAN,
             'it': Language.ITALIAN, 'en': Language.ENGLISH}


def read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def segment(text, detector, mapping):
    labels = ['UNKNOWN'] * len(text)
    previous_end = 0
    for hit in detector.detect_multiple_languages_of(text):
        start, end = hit.start_index, hit.end_index
        if not previous_end <= start < end <= len(text):
            raise ValueError('Invalid or overlapping detector offsets')
        labels[start:end] = [mapping[hit.language]] * (end - start)
        previous_end = end
    # Keep word-internal apostrophes and hyphens. Independent punctuation,
    # whitespace, symbols and standalone numbers are neutral.
    for i, char in enumerate(text):
        internal = (char in "'’‐‑-" and i > 0 and i + 1 < len(text)
                    and text[i-1].isalpha() and text[i+1].isalpha())
        if char.isspace() or (unicodedata.category(char)[0] in 'PS' and not internal):
            labels[i] = 'NEUTRAL'
    for match in re.finditer(r'\b\d+\b', text):
        labels[match.start():match.end()] = ['NEUTRAL'] * len(match.group())
    segments = []
    for char, lang in zip(text, labels):
        if segments and segments[-1]['lang'] == lang:
            segments[-1]['text'] += char
        else:
            segments.append({'text': char, 'lang': lang})
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--texts', required=True)
    parser.add_argument('--issues', required=True,
                        help='Extraction issues; matching sequence/record IDs are skipped')
    parser.add_argument('--source-language', choices=['fr', 'de', 'it', 'philippines'], required=True)
    parser.add_argument('--provisional-detector', choices=['tagalog'],
                        help='Explicit experimental proxy for unresolved Philippines source label')
    parser.add_argument('--limit', type=int, default=100,
                        help='Eligible unique texts; 0 processes all eligible texts')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.source_language != 'philippines' and args.provisional_detector:
        parser.error('--provisional-detector is only for the unresolved Philippines label')
    out = Path(args.out)
    if out.exists():
        parser.error('Output directory already exists; choose a new path')
    other = (Language.TAGALOG if args.source_language == 'philippines'
             else LANGUAGES[args.source_language])
    mapping = {other: args.source_language, Language.ENGLISH: 'en'}
    detector = LanguageDetectorBuilder.from_languages(other, Language.ENGLISH).build()
    issue_ids = {row['id'] for row in read_rows(args.issues)}
    texts, lids, seen_ids, seen_hashes = [], [], set(), set()
    skipped_issues = skipped_duplicates = 0
    tool_version = importlib.metadata.version('lingua-language-detector')
    source = f'lingua-language-detector/{tool_version}; mixed-language-high-accuracy'
    if args.provisional_detector:
        source += '; PROVISIONAL:TAGALOG-detector-to-unresolved-philippines-source-label'
    elif args.source_language == 'philippines':
        source += '; TAGALOG-to-philippines-source-label; SwitchLingua-prompt:Filipino(Tagalog)'
    for row in read_rows(args.texts):
        if set(row['languages']) != {args.source_language, 'en'}:
            raise ValueError('Input languages do not match --source-language')
        if row['sequence_id'] in issue_ids or row['record_id'] in issue_ids:
            skipped_issues += 1
            continue
        if row['sequence_id'] in seen_ids or row['text_sha256'] in seen_hashes:
            skipped_duplicates += 1
            continue
        if hashlib.sha256(row['text'].encode()).hexdigest() != row['text_sha256']:
            raise ValueError('Text checksum mismatch')
        lid = {'sequence_id': row['sequence_id'], 'text_sha256': row['text_sha256'],
               'annotation_source': source, 'review_status': 'unreviewed', 'reviewer': '',
               'segments': segment(row['text'], detector, mapping)}
        validate_lid(row, lid)
        texts.append(row)
        lids.append(lid)
        seen_ids.add(row['sequence_id'])
        seen_hashes.add(row['text_sha256'])
        if args.limit and len(texts) >= args.limit:
            break
    if not texts:
        raise ValueError('No eligible texts')
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [('texts.jsonl', texts), ('lid.jsonl', lids)]:
        with (out / name).open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'tool': 'lingua-language-detector', 'version': tool_version,
                'detector_languages': [other.name, Language.ENGLISH.name],
                'source_language': args.source_language,
                'language_mapping_note': ('Lingua TAGALOG maps to the preserved philippines source label. '
                                          'SwitchLingua core/prompt.py explicitly names Filipino (Tagalog). '
                                          'This documents the detector choice, not annotation accuracy.'
                                          if args.source_language == 'philippines' else None),
                'provisional_language_mapping': bool(args.provisional_detector),
                'sequences': len(texts), 'limit': args.limit,
                'selection': 'input order; skip extraction issues; deduplicate by ID and text hash',
                'skipped_issues_in_scanned_prefix': skipped_issues,
                'skipped_duplicates_in_scanned_prefix': skipped_duplicates,
                'inputs': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                           for p in [args.texts, args.issues]},
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'linguistically_verified': False,
                'limitations': ['Unreviewed heuristic pseudolabels; short switches and boundaries may be wrong.',
                                'No subword tokenization, predictive labels, split, or model evaluation performed.']}
    (out / 'lid_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(texts)} unreviewed sequences; structure/reconstruction only. Output: {out}')


if __name__ == '__main__':
    main()
