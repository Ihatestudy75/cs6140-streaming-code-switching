"""Prepare or verify a frozen data-v1.1 tokenizer (mBERT or XLM-R).

Everyone runs this instead of `pipeline.py prepare-tokenizer`, so all 14 pairs are
tokenized with byte-identical tokenizer files and the same `tokenizer_sha256`
(pipeline.py split rejects mixed hashes).

What it does
  1. Downloads the tokenizer at the pinned revision (pipeline.prepare_tokenizer).
  2. Windows writes the JSON/TXT files with CRLF line endings, which changes the hash.
     It converts only those text files to LF. Binary files (XLM-R's
     sentencepiece.bpe.model) are never modified: they contain CRLF byte pairs that
     are data, not line endings, and editing them corrupts the file.
  3. Recomputes the hash and stops with an error unless it equals the frozen value.
If the output folder already exists, it only verifies it and changes nothing.

Run from the repository root with the Update 1 environment (transformers 4.57.6):
  project_update_1/.venv/Scripts/python.exe project_update_2/prepare_tokenizer_v1_1.py --backbone xlmr   # Windows
  project_update_1/.venv/bin/python project_update_2/prepare_tokenizer_v1_1.py --backbone xlmr           # macOS/Linux
See DATA_V1.1.md.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'project_update_1'))
import pipeline as core  # noqa: E402

# Frozen data-v1.1 tokenizers. The mBERT entry is the Update 1 tokenizer, unchanged.
FROZEN = {
    'mbert': {'model': 'google-bert/bert-base-multilingual-cased',
              'revision': '3f076fdb1ab68d5b2880cb87a0886f315b8146f8',
              'tokenizer_sha256': '78b44c32238e3b5e83d1deaa63c3fa1466ddf73547ce641a36f75180079ab45f'},
    'xlmr': {'model': 'FacebookAI/xlm-roberta-base',
             'revision': 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089',
             'tokenizer_sha256': '088aef1538504d68661da8cf24bc0f98581c0e356cc7060ad724232d5f7545f2'},
}
TEXT_SUFFIXES = {'.json', '.txt'}  # everything else is treated as binary and left untouched


def verify(out, expected):
    """Return a list of problems; an empty list means the folder is the frozen tokenizer."""
    manifest_path = out / 'tokenizer_manifest.json'
    if not manifest_path.is_file() or not (out / 'tokenizer').is_dir():
        return [f'{out} is not a tokenizer folder (missing tokenizer/ or tokenizer_manifest.json)']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    actual = core.tree_hash(out / 'tokenizer')
    problems = [f'manifest {k} = {manifest.get(k)!r}, expected {expected[k]!r}'
                for k in ('model', 'revision', 'tokenizer_sha256') if manifest.get(k) != expected[k]]
    if actual != expected['tokenizer_sha256']:
        problems.append(f'hash of the tokenizer files = {actual}, expected {expected["tokenizer_sha256"]}')
    return problems


def prepare(out, expected):
    core.prepare_tokenizer(SimpleNamespace(model=expected['model'], revision=expected['revision'], out=str(out)))
    for p in sorted((out / 'tokenizer').rglob('*')):
        if p.is_file() and p.suffix in TEXT_SUFFIXES:
            data = p.read_bytes()
            if b'\r\n' in data:
                p.write_bytes(data.replace(b'\r\n', b'\n'))
    manifest = json.loads((out / 'tokenizer_manifest.json').read_text(encoding='utf-8'))
    manifest['tokenizer_sha256'] = core.tree_hash(out / 'tokenizer')
    core.write_json(out / 'tokenizer_manifest.json', manifest)


def main():
    p = argparse.ArgumentParser(description='Prepare or verify a frozen data-v1.1 tokenizer.')
    p.add_argument('--backbone', choices=sorted(FROZEN), required=True)
    p.add_argument('--out', help='tokenizer folder (default: project_update_1/assets/<backbone>)')
    a = p.parse_args()
    expected = FROZEN[a.backbone]
    out = Path(a.out) if a.out else ROOT / 'project_update_1' / 'assets' / a.backbone
    if out.exists():
        print(f'{out} exists: verifying only, nothing is changed.')
    else:
        prepare(out, expected)
    problems = verify(out, expected)
    if problems:
        sys.exit('FAIL: not the frozen data-v1.1 tokenizer.\n  ' + '\n  '.join(problems)
                 + '\nDelete the folder and run again. If it still fails, ask Yulun for the zipped folder.')
    print(f'PASS: frozen {a.backbone} tokenizer in {out}\n  sha256 {expected["tokenizer_sha256"]}')


if __name__ == '__main__':
    main()
