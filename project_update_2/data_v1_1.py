"""data-v1.1 tools: split all pairs, check that both tokenizers got the same split,
build the statistics table.

  split          run pipeline.py split once per backbone over the same pairs, pilot records forced
                 into train, then run compare-splits on the two results
  compare-splits check that the mBERT and XLM-R splits give every sequence the same split
  stats-table    per pair x backbone: sequences, positions, switches, S/M/L, UNKNOWN, split sizes
  export         copy the text-free files (stats, manifests, token issues, splits) into the repository

Input layout (DATA_V1.1.md §2): <root>/<backbone>/<pair>/{labeled_tokens.jsonl, stats.json}.
Outputs of split go to <root>/<backbone>/<name>/ and contain the original text (labeled_tokens.jsonl):
keep them local / private Drive. splits.jsonl, split_manifest.json and the stats table are text-free.

Run from the repository root with the Update 1 environment, e.g.
  PY=project_update_1/.venv/Scripts/python.exe
  $PY project_update_2/data_v1_1.py split --pilot-dir <folder with all pilot_record_ids files>
  $PY project_update_2/data_v1_1.py stats-table --out project_update_2/data_v1.1_stats
  $PY project_update_2/data_v1_1.py export --out project_update_2/data_v1.1
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'project_update_1'))
import pipeline as core  # noqa: E402

BACKBONES = ('mbert', 'xlmr')
LID_GROUP = {'ar_en': 'script_rule', 'zh_en': 'script_rule', 'ja_en': 'script_rule',
             'yue_en': 'script_rule', 'ko_en': 'script_rule', 'ru_en': 'script_rule',
             'es_en': 'llm', 'vi_en': 'llm', 'hi_en': 'llm', 'ms_en': 'llm',
             'fr_en': 'llm', 'de_en': 'llm', 'it_en': 'llm', 'philippines_en': 'llm'}  # relabeled with Claude in v1.1 (was Lingua)
PAIRS = list(LID_GROUP)
DEFAULT_ROOT = ROOT / 'project_update_1' / 'data' / 'v1.1'


def pair_dir(root, backbone, pair):
    return Path(root) / backbone / pair


def read_pilot_ids(pilot_dir):
    ids = []
    for p in sorted(Path(pilot_dir).glob('*.txt')):
        ids += [s.strip() for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(ids) != len(set(ids)):
        raise SystemExit('Duplicate record_id across pilot files')
    return ids


def cmd_split(a):
    pairs = a.pairs or PAIRS
    missing = [f'{b}/{p}' for b in BACKBONES for p in pairs
               if not (pair_dir(a.root, b, p) / 'labeled_tokens.jsonl').is_file()]
    if missing:
        raise SystemExit('Missing labeled_tokens.jsonl for: ' + ', '.join(missing))
    pilot = [rid for rid in read_pilot_ids(a.pilot_dir) if rid.split(':', 1)[0] in pairs]
    covered = {rid.split(':', 1)[0] for rid in pilot}
    if set(pairs) - covered:
        raise SystemExit('No pilot record ids for: ' + ', '.join(sorted(set(pairs) - covered)))
    for b in BACKBONES:
        out = Path(a.root) / b / a.name
        pilot_file = Path(a.root) / b / f'{a.name}_pilot_record_ids.txt'
        pilot_file.write_text('\n'.join(pilot) + '\n', encoding='utf-8')
        print(f'== {b}: {len(pairs)} pairs, {len(pilot)} pilot records forced into train')
        core.split(SimpleNamespace(inputs=[str(pair_dir(a.root, b, p) / 'labeled_tokens.jsonl') for p in pairs],
                                   seed=a.seed, train_records=str(pilot_file), out=str(out)))
    compare(Path(a.root) / 'mbert' / a.name, Path(a.root) / 'xlmr' / a.name)


def compare(dir_a, dir_b):
    sa = {r['sequence_id']: r for r in core.read_jsonl(Path(dir_a) / 'splits.jsonl')}
    sb = {r['sequence_id']: r for r in core.read_jsonl(Path(dir_b) / 'splits.jsonl')}
    problems = []
    if set(sa) != set(sb):
        problems.append(f'sequence ids differ: {len(set(sa) - set(sb))} only in {dir_a}, '
                        f'{len(set(sb) - set(sa))} only in {dir_b}')
    diff = [s for s in set(sa) & set(sb) if (sa[s]['split'], sa[s]['group_id']) != (sb[s]['split'], sb[s]['group_id'])]
    if diff:
        problems.append(f'{len(diff)} sequences have a different split or group, e.g. {sorted(diff)[:3]}')
    counts = Counter(r['split'] for r in sa.values())
    if problems:
        raise SystemExit('FAIL: the two splits differ\n  ' + '\n  '.join(problems))
    print(f'PASS: identical splits for {len(sa)} sequences {dict(sorted(counts.items()))}')


def cmd_compare(a):
    compare(a.a, a.b)


def cmd_stats(a):
    split_counts = {}
    for b in BACKBONES:
        mf = Path(a.root) / b / a.name / 'split_manifest.json'
        if mf.is_file():
            split_counts[b] = json.loads(mf.read_text(encoding='utf-8'))['by_pair']
    rows = []
    for p in PAIRS:
        for b in BACKBONES:
            f = pair_dir(a.root, b, p) / 'stats.json'
            if not f.is_file():
                continue
            s = json.loads(f.read_text(encoding='utf-8'))
            d = s['duration_class_proportions']
            sc = split_counts.get(b, {}).get(p, {})
            rows.append({'pair': p, 'lid_group': LID_GROUP[p], 'backbone': b, 'sequences': s['sequences'],
                         'positions': s['valid_positions'], 'switches': s['switches'],
                         'switch_rate': round(s['switch_rate'], 4),
                         'S': round(d['0'], 4), 'M': round(d['1'], 4), 'L': round(d['2'], 4),
                         'switches_without_duration': s['switches_without_duration'],
                         'unknown_subwords': s['unknown_subwords'],
                         'train': sc.get('train', ''), 'validation': sc.get('validation', ''), 'test': sc.get('test', '')})
    if not rows:
        raise SystemExit(f'No stats.json found under {a.root}')
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.parent / (out.name + '.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pct = lambda x: f'{x:.1%}'  # noqa: E731
    lines = ['| Pair | LID | Tokenizer | Seqs | Positions | Switches | Rate | S / M / L | No dur. | UNK | Train / Val / Test |',
             '|---|---|---|---:|---:|---:|---:|---|---:|---:|---|']
    for r in rows:
        tvt = f'{r["train"]} / {r["validation"]} / {r["test"]}' if r['train'] != '' else '–'
        lines.append(f'| {r["pair"]} | {r["lid_group"]} | {r["backbone"]} | {r["sequences"]:,} | {r["positions"]:,} | '
                     f'{r["switches"]:,} | {pct(r["switch_rate"])} | {pct(r["S"])} / {pct(r["M"])} / {pct(r["L"])} | '
                     f'{r["switches_without_duration"]} | {r["unknown_subwords"]} | {tvt} |')
    (out.parent / (out.name + '.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    pairs_found = sorted({r['pair'] for r in rows})
    print(f'Wrote {out.parent / (out.name + ".csv")} and {out.parent / (out.name + ".md")}: {len(pairs_found)} pairs, {len(rows)} rows')
    if len(pairs_found) < len(PAIRS):
        print('  missing pairs:', ', '.join(p for p in PAIRS if p not in pairs_found))

# Keys that would carry dataset text; export refuses any JSON object that has one.
TEXT_KEYS = {'text', 'segments', 'tokens', 'words'}


def relative_paths(value):
    """Rewrite absolute paths inside the repository as repo-relative POSIX paths."""
    if isinstance(value, dict):
        return {relative_paths(k): relative_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relative_paths(v) for v in value]
    if isinstance(value, str):
        for root in {str(ROOT), str(ROOT).replace('\\', '/')}:
            if root in value:
                value = value.replace(root + '\\', '').replace(root + '/', '').replace(root, '.')
                value = value.replace('\\', '/')
        return value
    return value


def assert_text_free(obj, where):
    if isinstance(obj, dict):
        bad = TEXT_KEYS & set(obj)
        if bad:
            raise SystemExit(f'{where}: refuses to export text field(s) {sorted(bad)}')
        for v in obj.values():
            assert_text_free(v, where)
    elif isinstance(obj, list):
        for v in obj:
            assert_text_free(v, where)


def export_file(src, dst):
    """Copy one text-free file; JSON/JSONL are checked for text keys and get repo-relative paths."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == '.json':
        obj = relative_paths(json.loads(src.read_text(encoding='utf-8')))
        assert_text_free(obj, src)
        core.write_json(dst, obj)
    elif src.suffix == '.jsonl':
        rows = [relative_paths(r) for r in core.read_jsonl(src)]
        for r in rows:
            assert_text_free(r, src)
        core.write_jsonl(dst, rows)
    else:
        dst.write_bytes(src.read_bytes().replace(b'\r\n', b'\n'))


def cmd_export(a):
    """<root>/<bb>/<pair>/{stats.json, manifest.json, token_issues.jsonl} for every pair and
    <root>/<bb>/<name>/{splits.jsonl, split_manifest.json} -> <out>/<bb>/..."""
    out = Path(a.out)
    n = 0
    for b in BACKBONES:
        for p in PAIRS:
            for f in ('stats.json', 'manifest.json', 'token_issues.jsonl'):
                export_file(pair_dir(a.root, b, p) / f, out / b / p / f)
                n += 1
        for f in ('splits.jsonl', 'split_manifest.json'):
            export_file(Path(a.root) / b / a.name / f, out / b / a.name / f)
            n += 1
    print(f'Exported {n} text-free files -> {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('split', help='split all pairs for both backbones, then compare')
    s.add_argument('--root', default=str(DEFAULT_ROOT))
    s.add_argument('--pilot-dir', required=True, help='folder with every pair\'s pilot_record_ids .txt file')
    s.add_argument('--pairs', nargs='+', choices=PAIRS, help='default: all 14 pairs')
    s.add_argument('--name', default='combined', help='output folder name under <root>/<backbone>/')
    s.add_argument('--seed', type=int, default=6140)
    s.set_defaults(func=cmd_split)
    s = sub.add_parser('compare-splits', help='check that two split folders assign every sequence identically')
    s.add_argument('--a', required=True)
    s.add_argument('--b', required=True)
    s.set_defaults(func=cmd_compare)
    s = sub.add_parser('stats-table', help='statistics table for every pair x backbone found')
    s.add_argument('--root', default=str(DEFAULT_ROOT))
    s.add_argument('--name', default='combined', help='split folder to read split sizes from (if present)')
    s.add_argument('--out', required=True, help='output path without extension; writes .csv and .md')
    s.set_defaults(func=cmd_stats)
    s = sub.add_parser('export', help='copy the text-free files into the repository')
    s.add_argument('--root', default=str(DEFAULT_ROOT))
    s.add_argument('--name', default='combined')
    s.add_argument('--out', required=True, help='e.g. project_update_2/data_v1.1')
    s.set_defaults(func=cmd_export)
    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
