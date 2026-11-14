# Word-level LID instructions, v1.1 addendum (fr, de, it, Tagalog)

Used with [`project_update_1/claude_lid_instructions.md`](../project_update_1/claude_lid_instructions.md)
to relabel `fr_en`, `de_en`, `it_en` and `philippines_en` for data-v1.1 (`relabel_v1_1.py`).
The input, the output and rules 1–6 there apply unchanged. **This file adds no new rule.** It
only gives examples of rules 1–6 (and of `LID_PROMPT.md` v1.0) for French, German, Italian and
Tagalog, so that these four pairs are labeled to the same standard as es_en, vi_en and hi_en.

The native language (`n`) of each batch: `fr_en` French · `de_en` German · `it_en` Italian ·
`philippines_en` Tagalog/Filipino.

## 7. Mixed words (rule 5 and `LID_PROMPT.md` rule 3)

A word (as listed in `words`; hyphenated forms are one word) gets one code.

- **Native inflection or affix on an English stem → `n`** (rule 5: "native inflection").
  - Tagalog: `nag-shopping`, `i-check`, `mag-work`, `nagda-drive`, `na-stress`, `mag-download`,
    `Magda-download`, `ini-explain`, `pinost`, `nag-text`
  - German: `gecheckt`, `downgeloadet`, `gechillt`, `chillen`, `gepostet`
  - French: `checker`, `liker`, `googler`, `briefé`
  - Italian: `chattare`, `postato`, `scrollando`, `googlare`
- **A compound that joins an English part and a native part, with no native inflection settling
  it → `u`** (`LID_PROMPT.md` rule 3: mixed words that cannot be assigned reliably are UNKNOWN):
  German `Smartphone-Trends`, `Social-Media-Strategie` → `u`. A word that is English as a whole
  (`real-life`, `long-term`) is `e`; a word that is native as a whole is `n`.
- **An English word with no native affix stays `e`**, even inside a native sentence:
  Tagalog `Medyo complicated ang situation` → `n e n e`.

## 8. Language notes (rules 1 and 4)

- **German:** every noun is capitalized. A capital letter alone does not make a word a proper
  noun. `Menschen`, `Woche` → `n`. Proper nouns follow rule 4 as in the other pairs.
- **Tagalog:**
  - Function words are native even when they look like English: `ang`, `ng`, `sa`, `na`, `at`,
    `ay`, `mga`, `ni`, `si`, `ko`, `mo`, `po`, `pa`, `ba`, `din`/`rin`, `lang`, `yung`, `kasi`,
    `grabe`, `talaga`, `sobra`.
  - `at` ("and") and `may` ("there is") are Tagalog in a Tagalog clause, and English
    (`at home`, `may be`) in an English clause.
  - Spanish-origin words that are ordinary Tagalog are `n`: `pero`, `siguro`, `medyo`, `pwede`,
    `syempre`.
- **French, Italian:** the elided forms `l'`, `d'`, `qu'`, `c'est`, `dell'`, `un'` are part of the
  word they attach to (rule 3) → `n`.
- **Short homographs** take the language of the clause they are in (rule 1):
  - French: `a`, `en`, `on`, `me`, `son`, `or`, `as`
  - German: `in`, `an`, `so`, `was`, `will`, `man`, `also`, `die`, `am`, `war`, `hat`
  - Italian: `a`, `e`, `i`, `o`, `in`, `no`, `come`, `se`, `me`, `era`

## 9. Loanwords (rule 5, examples for these languages)

- **Established native words → `n`.** These are in the native dictionary and spelled natively,
  or have been ordinary vocabulary for a long time:
  - French: `week-end`, `parking`, `football`, `sport`, `film`, `internet`
  - German: `Handy`, `Computer`, `Internet`, `Sport`, `Film`, `Hotel`
  - Italian: `sport`, `film`, `computer`, `internet`, `bar`
- **English kept as English → `e`.** These are switches in the generated text: `seriously`,
  `maybe`, `deadline`, `meeting`, `cool`, `smartphone`, `trend`, `team`, `coach`, `fans`, `show`.
- When it is unclear, use the clause rule (rule 5, third bullet). If that does not settle it either,
  use `u`.
