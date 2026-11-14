# Harmonization policy (data-v1.1 relabel: fr_en, de_en, it_en, philippines_en)

> This is the policy given verbatim to the four harmonizers and the four pilot reviewers
> (see [`RELABEL_V1_1.md`](RELABEL_V1_1.md)). The precedent tables named in section A
> (`<pair>_decisions_by_word.tsv`) were built locally from the Update 1 harmonization
> candidates and decisions of es_en, vi_en and hi_en; they list dataset words, so they are not
> committed.

You are the single harmonizer for ONE pair. Several parallel labelers coded every word n / e / u.
`candidates.jsonl` lists the occurrences that need one consistent pass:
- every `u`
- capitalized words that do not start the item (for de_en only when coded e/u, or when the same
  capitalized word got different codes)
- an `e` word between two `n` words
- every occurrence of a lowercase word that got both e and n somewhere
- loanword audit: a word coded `n` that is coded `e` at least 3 times elsewhere in the four pairs

Each candidate row: {"cid", "id", "k" (word index in the item), "word", "current", "text"}.
Write a decision for EVERY cid: {"cid": <cid>, "label": "n"|"e"|"u"} (one JSON line each).

The rules are the same as the labelers': project_update_1/claude_lid_instructions.md (rules 1–6)
plus project_update_2/claude_lid_instructions_v1_1.md (rules 7–9). They are the Update 1 rules
used for es_en, vi_en and hi_en. This policy fixes the points where the labelers disagreed.

## A. Precedent from Update 1 (keep the same standard)
`es_en_decisions_by_word.tsv`, `vi_en_…`, `hi_en_…` in this folder are the final Update 1 decisions.
For a name, brand or acronym that is not of native origin for your pair, use the majority code
there. Examples: Trump u · Musk u · Google u · Samsung u · TikTok u · Amazon u · Texas u ·
China u · Apple e · Microsoft e · Instagram e · Netflix e · NASA e · AI e · CEO e · COVID e ·
Washington e · Taylor Swift e e.

## B. Names (rule 4), relative to the pair
Update 1 judged a name's origin relative to the pair (`Kamala` was u in es_en but hi in hi_en).
- A name of native origin or native form for this pair → n, even if English spells it the same:
  fr `Macron`, `Marseille`, `Paris`; de `Berlin`, `München`, `Müller`, `Bayern`;
  it `Italia`, `Juventus`, `Roma`; philippines `Pilipinas`, `Marcos`, `Duterte`, `Juan`, `Reyes`.
  Decide by: "Is this the native-language form of a name from the native-language area?" → n.
  When the text uses the English form of such a name (`Munich`, `Rome`, `Philippines`) → e.
  `France` is spelled the same in French and English and is the native name → n in fr_en.
- An English-language name or form → e (`Liverpool`, `Jake`, `Premier League`, `Filipino`).
- A name from a third language, or a common given name shared by many languages (`David`,
  `Thomas`, `Maria`, `Laura`) → u.
- Possessive: English `'s` on a name → judge the name; if the name alone would be n or u, the
  word is mixed → u; if the name alone is e → e. Native genitive/elision (`Trumps`, `d'Amazon`,
  `l'FBI`) → n (rule 7).
- The same name must get the same code everywhere in the pair, unless the form differs.

## C. Loanwords (rule 5) — this is where the labelers disagreed most
An English-origin word in English spelling, with no native inflection or native spelling, is `e`
even inside a native clause. This matches Update 1 es_en (`fans`, `show`, `chat`, `gadgets`,
`smartphone` = e although Spanish dictionaries list them). Therefore e:
team, fan(s), show, meeting, interview, design, leader, manager, management, partner, business,
weekend, budget, marketing, challenge, stress, ticket(s), match, event, update(s), feedback, job,
trend, coach, online, app, smartphone, privacy, look, cast, sponsor, influencer, follower,
playoff, derby, layout, timing, leadership, cookies, startup, brunch, party, laptop, display, …
Only these very old, fully nativized loans are `n` inside a native clause: sport, film, club,
hotel, golf, tennis, internet, computer, robot, video, radio, bar, tv, drama, goal (it),
week-end (fr spelling), Handy (de). Plus any word whose spelling is an ordinary native word of
non-English origin (fr `important`, `situation`, `nation`, `article`, `attention`; Tagalog
`normal`, `simple`, `regular`, `hospital` in a Tagalog clause) → clause language (rule 5).
A native-inflected or natively spelled form is n (French plurals `coachs`, `matchs`; German
genitive `des Teams`; `coole`, `gecheckt`, `nag-shopping`, Tagalog spelling `basketbol`).
In an English clause, all of these are e.

## D. Shifted labels
Many candidates are frequent function words (`de`, `ng`, `ang`, `the`, `you`, `ist`, `il`,
`per`) flagged because one occurrence somewhere got the wrong code (often a labeler's run was off
by one word). Check every occurrence in context: a native function word in native text is n,
an English one is e. Fix the wrong ones. When you find a shifted occurrence, its neighbours in
the same item are probably shifted too; if they are not candidates, list the item id in your
final report (do not edit files other than your decisions file).

## E. Mixed and third-language words
Mixed compounds with no native inflection → u (rule 7). Third languages (Spanish `podemos`,
Hiligaynon `kag`, Japanese `anime`) → u. Acronyms: native-language acronym → n, English → e,
neither or letters/model codes → u.
