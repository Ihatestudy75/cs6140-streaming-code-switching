"""Run provenance and the tokenizer/backbone guard.

check_tokenizer: every data row stores `tokens`, made by pipeline.py with
tokenizer.convert_ids_to_tokens(input_ids). Converting the same input_ids with the
backbone's tokenizer must give the same tokens. If not, the data was tokenized
for a different backbone (mBERT IDs are all valid XLM-R IDs, so without this check
training would run on wrong inputs without any error).
"""
import json
import hashlib
import subprocess
from pathlib import Path


def check_tokenizer(data_path, tokenizer, backbone):
    """Read the data file directly (load_rows drops `tokens` on purpose) and
    check every row's input_ids against its stored tokens under this tokenizer."""
    hashes = set()
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            hashes.add(r['tokenizer_sha256'])
            if tokenizer.convert_ids_to_tokens(r['input_ids']) != r['tokens']:
                raise SystemExit(f"{r['sequence_id']}: input_ids do not match the stored tokens under the "
                                 f"{backbone} tokenizer; the data was tokenized for a different backbone.")
    return hashes.pop()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=Path(__file__).parent,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None