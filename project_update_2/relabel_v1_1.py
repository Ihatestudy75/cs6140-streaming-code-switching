#!/usr/bin/env python3
"""Relabel fr_en, de_en, it_en and philippines_en with Claude for data-v1.1 (Yulun).

Update 1 labelled these four pairs with Lingua and screened out sequences with likely errors;
the 500 records came from the screened pool. Lingua missed most short English insertions, so for
v1.1 the four pairs are rebuilt with the same procedure as the other LLM pairs (es_en, vi_en):
full extraction -> 500 records sampled from the WHOLE CSV (seed '6140:<pair>') -> word-level
Claude LID (annotate_claude.py) -> model-assisted review of 20 pilot records -> tokenize.

Run from the repository root with the Update 1 venv. Outputs contain dataset text and go to
project_update_1/data/ (ignored by Git). No Git operations.

  extract    --csv-dir <folder with the four CSVs>
             full extraction, 500-record sample, pilot ids, cs_type map per pair
  (prepare)  annotate_claude.py prepare --texts <pair>/extraction/texts.jsonl --out <pair>/claude_work
  (label)    Claude subagents write claude_work/batch_NNN.labels.jsonl (claude_lid_instructions.md +
             project_update_2/claude_lid_instructions_v1_1.md)
  collect    harmonization candidates -> <pair>/candidates.jsonl
  (decide)   one harmonizer per pair writes <pair>/harmonize_decisions.jsonl; corrections found
             afterwards go to <pair>/extra_fixes.jsonl
  build-lid  decisions + extra fixes -> annotate_claude apply -> assemble -> <pair>/annotation/
  (review)   model-assisted review of the 20 pilot records -> <pair>/review_decisions.jsonl
  finalize   apply the review, write final lid + lid_offsets + review.csv, tokenize with mBERT and
             XLM-R into project_update_1/data/v1.1/<backbone>/<pair>/
  export     text-free records -> project_update_2/data_v1.1/relabel/<pair>/

The steps in parentheses were done in Claude Code and cannot run unattended; their outputs
(labels, decisions) are kept as character offsets, so lid_offsets.jsonl is the official version.
Options go before the subcommand: relabel_v1_1.py --pairs fr_en de_en --work <dir> build-lid
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'project_update_1'))
import pipeline as core  # noqa: E402
import run_unicode_claude as runner  # noqa: E402

# pair -> (CSV file, native code used by Yufan in Update 1, CSV sha256)
PAIRS = {
    'fr_en': ('French_eng.csv', 'fr', '62970a8adc631380774045bf13478bb7b059508f73caef218da50ee060f238d0'),
    'de_en': ('German_Eng.csv', 'de', '8e3f9a07c453555e7ead719bb8352dd56992330730914a6fe123c563aaa7c8f3'),
    'it_en': ('Italian_eng.csv', 'it', 'd76eac6442a157187b57069eeba2553205af7d82473da821f58ae54d393a1767'),
    'philippines_en': ('Philippines_eng.csv', 'philippines', 'b111cae514ec95ba29df66f9cae9df341e8e855830753ec3b51f8bb6d328cca2'),
}
WORK = ROOT / 'project_update_1' / 'data' / 'v1.1_relabel'
SEED, RECORDS, PILOT = 6140, 500, 20

# Short function words that are true homographs of English words; annotate_claude.HOMOGRAPHS is
# the Spanish list. Same purpose: do not send every occurrence of these to harmonization.
HOMOGRAPHS = {
    'fr_en': {'a', 'en', 'on', 'me', 'son', 'or', 'as', 'i'},
    'de_en': {'in', 'an', 'so', 'was', 'will', 'man', 'also', 'die', 'am', 'war', 'hat', 'i'},
    'it_en': {'a', 'e', 'i', 'o', 'in', 'no', 'come', 'se', 'me', 'era'},
    'philippines_en': {'at', 'may', 'man', 'to', 'na', 'pa', 'sa', 'ba', 'di', 'i', 'a'},
}
# German capitalizes every noun, so "capitalized and not item-initial" would flag most German
# nouns. For de_en a capitalized word is a candidate only if it is coded e/u, or if the same
# capitalized word received different codes somewhere in the pair.
CAPITAL_NATIVE_EXEMPT = {'de_en'}


def cmd_extract(args):
    csv_dir = Path(args.csv_dir).resolve()
    work = Path(args.work).resolve()
    for pair in args.pairs:
        filename, code, expected = PAIRS[pair]
        raw = csv_dir / filename
        runner.check(runner.sha(raw) == expected, f'{raw}: sha256 differs from the Update 1 file')
        base = work / pair
        full = base / 'extraction_full'
        core.extract(SimpleNamespace(input=str(raw), languages=f'{code},en', limit=0, out=str(full)))
        texts = runner.sample(core, full, base / 'extraction', pair, SEED, RECORDS)
        record_ids = sorted({t['record_id'] for t in texts})
        pilot = sorted(random.Random(f'{SEED}:{pair}:pilot').sample(record_ids, min(PILOT, len(record_ids))))
        (base / 'pilot_record_ids.txt').write_text(''.join(r + '\n' for r in pilot), encoding='utf-8', newline='\n')
        core.write_jsonl(base / 'cs_type.jsonl', runner.cs_types(raw, texts))
        print(f'{pair}: {len(pilot)} pilot records', flush=True)


def cmd_collect(args):
    """annotate_claude.collect with a per-language homograph list (and the German capital rule)."""
    import annotate_claude as ac
    # Loanword audit: a word coded n that is coded e at least 3 times across the four pairs.
    # Without it, a labeler who always codes e.g. "interview" as native is never re-checked.
    english = Counter()
    for pair in PAIRS:
        for r in ac.load(Path(args.work).resolve() / pair / 'claude_work').values():
            english.update(w.lower() for w, c in zip(r['words'], r['labels']) if c == 'e')
    english = {w for w, k in english.items() if k >= 3}
    for pair in args.pairs:
        base = Path(args.work).resolve() / pair
        items = ac.load(base / 'claude_work')
        codes = defaultdict(set)
        for r in items.values():
            for w, c in zip(r['words'], r['labels']):
                codes[w].add(c)
        mixed = {w for w, cs in codes.items() if w.islower() and {'e', 'n'} <= cs and w not in HOMOGRAPHS[pair]}
        cap_mixed = {w for w, cs in codes.items() if not w.islower() and len(cs) > 1}
        rows = []
        for r in items.values():
            w, c = r['words'], r['labels']
            for k in range(len(w)):
                insertion = c[k] == 'e' and 0 < k < len(w) - 1 and c[k - 1] == 'n' and c[k + 1] == 'n'
                capital = k > 0 and w[k][:1].isupper() and w[k] != 'I'
                if capital and pair in CAPITAL_NATIVE_EXEMPT:
                    capital = c[k] != 'n' or w[k] in cap_mixed
                loan = c[k] == 'n' and w[k].lower() in english and w[k].lower() not in HOMOGRAPHS[pair]
                if c[k] == 'u' or capital or insertion or w[k] in mixed or loan:
                    rows.append({'cid': len(rows) + 1, 'id': r['id'], 'k': k, 'word': w[k],
                                 'current': c[k], 'text': r['text']})
        out = base / 'candidates.jsonl'
        runner.check(not out.exists(), f'{out} exists')
        core.write_jsonl(out, rows)
        print(f'{pair}: {len(rows)} candidates ({len({x["word"] for x in rows})} word types) -> {out}', flush=True)


def cmd_build_lid(args):
    """harmonize decisions + extra fixes -> annotate_claude apply -> assemble -> <pair>/annotation/.

    extra_fixes.jsonl ({id, k, label, reason}) holds corrections found after collect: a shifted
    word next to a fixed candidate that was not itself a candidate, and cross-pair checks. A fix
    on an existing candidate overrides its decision; any other fix is appended as a new candidate,
    so harmonize_changes.jsonl records it like every other change.
    """
    import annotate_claude as ac
    for pair in args.pairs:
        base = Path(args.work).resolve() / pair
        work = base / 'claude_work'
        items = ac.load(work)
        cands = core.read_jsonl(base / 'candidates.jsonl')
        decisions = {r['cid']: r['label'] for r in core.read_jsonl(base / 'harmonize_decisions.jsonl')}
        runner.check(set(decisions) == {c['cid'] for c in cands}, f'{pair}: decisions do not cover the candidates')
        fixes = core.read_jsonl(base / 'extra_fixes.jsonl') if (base / 'extra_fixes.jsonl').stat().st_size else []
        where = {(c['id'], c['k']): c['cid'] for c in cands}
        overridden = appended = 0
        for f in fixes:
            r = items[f['id']]
            runner.check(0 <= f['k'] < len(r['words']) and f['label'] in 'neu', f'{pair}: bad fix {f}')
            if (f['id'], f['k']) in where:
                decisions[where[f['id'], f['k']]] = f['label']
                overridden += 1
            else:
                cid = len(cands) + 1
                cands.append({'cid': cid, 'id': f['id'], 'k': f['k'], 'word': r['words'][f['k']],
                              'current': r['labels'][f['k']], 'text': r['text']})
                decisions[cid] = f['label']
                appended += 1
        core.write_jsonl(base / 'candidates_final.jsonl', cands)
        core.write_jsonl(base / 'decisions_final.jsonl', [{'cid': c, 'label': decisions[c]} for c in sorted(decisions)])
        final = base / 'claude_work_final'
        ac.apply(SimpleNamespace(work=str(work), candidates=str(base / 'candidates_final.jsonl'),
                                 decisions=str(base / 'decisions_final.jsonl'), out=str(final)))
        ann = base / 'annotation'
        ac.assemble(SimpleNamespace(texts=str(base / 'extraction' / 'texts.jsonl'), work=str(final),
                                    model='claude-opus-5-5', script_native=False, out=str(ann)))
        manifest = json.loads((ann / 'lid_manifest.json').read_text(encoding='utf-8'))
        core.write_json(ann / 'lid_manifest.json', dict(
            manifest, interface='Claude Code (interactive session, subagents: 2 labelers + 1 harmonizer per pair)',
            prompt_addendum='project_update_2/claude_lid_instructions_v1_1.md',
            prompt_addendum_sha256=runner.sha(ROOT / 'project_update_2' / 'claude_lid_instructions_v1_1.md'),
            harmonize={'candidates': len(cands), 'extra_fixes': len(fixes), 'fixes_overriding_decisions': overridden,
                       'fixes_appended': appended,
                       'changes': len(core.read_jsonl(final / 'harmonize_changes.jsonl'))},
            relabel_script_sha256=runner.sha(Path(__file__))))
        print(f'{pair}: {overridden} fixes overrode decisions, {appended} appended', flush=True)


REVIEWER = 'claude-opus-5-5 (model-assisted review, delegated by yulun)'
CODE = {'e': 'en', 'u': 'UNKNOWN'}


def cmd_finalize(args):
    """Pilot review -> final lid -> tokenize with both backbones into data/v1.1/<bb>/<pair>/.

    review_decisions.jsonl: one line per pilot item {sequence_id, fixes: [{k, label, issue_type,
    reason}]}. The fixes are applied word by word; every pilot item becomes `reviewed`. review.csv
    keeps only ids, character offsets and codes (no text), so it can be committed.
    """
    import annotate_claude as ac
    import datetime
    import csv
    today = datetime.date.today().isoformat()
    for pair in args.pairs:
        base = Path(args.work).resolve() / pair
        code = PAIRS[pair][1]
        lang = {'n': code, **CODE}
        texts = core.read_jsonl(base / 'annotation' / 'texts.jsonl')
        lids = {r['sequence_id']: r for r in core.read_jsonl(base / 'annotation' / 'lid.jsonl')}
        pilot = set((base / 'pilot_record_ids.txt').read_text(encoding='utf-8').split())
        pilot_items = [t['sequence_id'] for t in texts if t['record_id'] in pilot]
        review = core.read_jsonl(base / 'review_decisions.jsonl')
        runner.check([r['sequence_id'] for r in review] == pilot_items, f'{pair}: review does not cover the pilot items in order')
        by_seq = {r['sequence_id']: r['fixes'] for r in review}
        rows, nfix = [], 0
        out_lids = []
        for t in texts:
            lid = lids[t['sequence_id']]
            if t['sequence_id'] in by_seq:
                chars = [s['lang'] for s in lid['segments'] for _ in s['text']]
                spans = ac.words(t['text'])
                for f in by_seq[t['sequence_id']]:
                    a, z = spans[f['k']]
                    before = chars[a]
                    runner.check(f['label'] in lang and lang[f['label']] != before, f'{t["sequence_id"]}: bad fix {f}')
                    chars[a:z] = [lang[f['label']]] * (z - a)
                    rows.append([t['sequence_id'], REVIEWER, today, 'reviewed', f['issue_type'], before,
                                 lang[f['label']], f'word {f["k"]} chars {a}-{z}'])
                    nfix += 1
                if not by_seq[t['sequence_id']]:
                    rows.append([t['sequence_id'], REVIEWER, today, 'reviewed', 'none', '', '', 'no change'])
                segs = []
                for ch, lab in zip(t['text'], chars):
                    if segs and segs[-1]['lang'] == lab:
                        segs[-1]['text'] += ch
                    else:
                        segs.append({'text': ch, 'lang': lab})
                lid = dict(lid, review_status='reviewed', reviewer=REVIEWER, segments=segs)
            core.validate_lid(t, lid)
            out_lids.append(lid)
        final = base / 'final'
        final.mkdir()
        core.write_jsonl(final / 'texts.jsonl', texts)
        core.write_jsonl(final / 'lid.jsonl', out_lids)
        core.write_jsonl(final / 'lid_offsets.jsonl', [runner.to_offsets(l) for l in out_lids])
        with (final / 'review.csv').open('w', encoding='utf-8', newline='') as handle:
            w = csv.writer(handle)
            w.writerow(['sequence_id', 'reviewer', 'review_date', 'status', 'issue_type', 'before', 'after', 'reason'])
            w.writerows(rows)
        core.write_json(final / 'build_manifest.json', {
            'lid_source': 'Claude word-level LID (annotate_claude.py + claude_lid_instructions_v1_1.md), harmonized',
            'items': len(out_lids), 'reviewed_items': len(pilot_items), 'review_rows': len(rows), 'fixes': nfix,
            'pilot_rule': "random.Random(f'{seed}:{pair}:pilot').sample(sorted record_ids with items, 20)",
            'relabel_script_sha256': runner.sha(Path(__file__))})
        for bb in ('mbert', 'xlmr'):
            out = ROOT / 'project_update_1' / 'data' / 'v1.1' / bb / pair
            core.tokenize(SimpleNamespace(texts=str(final / 'texts.jsonl'), lid=str(final / 'lid.jsonl'),
                                          tokenizer_dir=str(ROOT / 'project_update_1' / 'assets' / bb),
                                          neutral_policy='inherit_previous', end_policy='observed',
                                          allow_unreviewed=True, out=str(out)))
        print(f'{pair}: {len(pilot_items)} pilot items reviewed, {nfix} fixes; tokenized mbert + xlmr', flush=True)


def cmd_export(args):
    """Text-free relabel records -> <out>/<pair>/ (repository). Uses data_v1_1's text guard."""
    sys.path.insert(0, str(ROOT / 'project_update_2'))
    import data_v1_1 as dv
    out = Path(args.out)
    for pair in args.pairs:
        base = Path(args.work).resolve() / pair
        dst = out / pair
        runner.check(not dst.exists(), f'{dst} exists')
        files = {
            'sampled_record_ids.jsonl': base / 'extraction' / 'sampled_record_ids.jsonl',
            'pilot_record_ids.txt': base / 'pilot_record_ids.txt',
            'extract_manifest.json': base / 'extraction' / 'extract_manifest.json',
            'extract_issues.jsonl': base / 'extraction' / 'extract_issues.jsonl',
            'cs_type.jsonl': base / 'cs_type.jsonl',
            'prepare_manifest.json': base / 'claude_work' / 'prepare_manifest.json',
            'harmonize_changes.jsonl': base / 'claude_work_final' / 'harmonize_changes.jsonl',
            'harmonize_manifest.json': base / 'claude_work_final' / 'harmonize_manifest.json',
            'extra_fixes.jsonl': base / 'extra_fixes.jsonl',
            'lid_manifest.json': base / 'annotation' / 'lid_manifest.json',
            'lid_offsets.jsonl': base / 'final' / 'lid_offsets.jsonl',
            'review.csv': base / 'final' / 'review.csv',
            'build_manifest.json': base / 'final' / 'build_manifest.json',
        }
        for name, src in files.items():
            dv.export_file(src, dst / name)
        # sequence index: every texts.jsonl field except the text itself
        rows = [{k: v for k, v in t.items() if k != 'text'} for t in core.read_jsonl(base / 'final' / 'texts.jsonl')]
        for r in rows:
            dv.assert_text_free(r, 'sequence_index')
        core.write_jsonl(dst / 'sequence_index.jsonl', rows)
        print(f'{pair}: {len(files) + 1} text-free files -> {dst}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--work', default=str(WORK))
    p.add_argument('--pairs', nargs='+', choices=list(PAIRS), default=list(PAIRS))
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('extract')
    s.add_argument('--csv-dir', required=True)
    s.set_defaults(func=cmd_extract)
    s = sub.add_parser('collect', help='harmonization candidates -> <pair>/candidates.jsonl')
    s.set_defaults(func=cmd_collect)
    s = sub.add_parser('build-lid', help='decisions + extra fixes -> apply -> assemble -> <pair>/annotation/')
    s.set_defaults(func=cmd_build_lid)
    s = sub.add_parser('finalize', help='pilot review -> final lid -> tokenize both backbones into data/v1.1/')
    s.set_defaults(func=cmd_finalize)
    s = sub.add_parser('export', help='text-free relabel records -> repository')
    s.add_argument('--out', default=str(ROOT / 'project_update_2' / 'data_v1.1' / 'relabel'))
    s.set_defaults(func=cmd_export)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
