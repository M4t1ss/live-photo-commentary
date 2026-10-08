# Data files

Large files are compressed with `xz -9e` (`.xz`); the crate decompresses them
when first used. To edit one, run `xz -d FILE.xz`, then `xz -9e FILE`.

- `us_heteronyms.json`, `gb_heteronyms.json`: the part-of-speech-dependent
  entries of `us_gold.json` and `gb_gold.json` from
  [misaki](https://github.com/hexgrad/misaki) 0.9.4 (Apache-2.0). Only
  lowercase words whose variants are keyed by content-word tags (`NOUN`,
  `VERB`, `ADJ`, `ADV`, `VBD`, `VBN`, `VBP`) are kept.
- `ja_words.txt.xz`: misaki 0.9.4's Japanese word list (Apache-2.0), used to
  group morphemes into words for spacing. Change from misaki: compressed
  with xz; the contents are unchanged.
- `tagger/`: weights for NLTK's averaged perceptron part-of-speech tagger
  (Penn Treebank tags), as converted to JSON by
  [misaki-rs](https://github.com/MicheleYin/misaki-rs) (MIT, © 2026 Michele Yin).
  The model itself is NLTK's `averaged_perceptron_tagger`, from Matthew
  Honnibal's textblob-aptagger and distributed with NLTK (Apache-2.0). It was
  trained on the Wall Street Journal part of the Penn Treebank, an LDC corpus
  with its own licence. Like NLTK and many other projects, this crate
  redistributes the trained weights, but not the corpus. Change from
  misaki-rs: `weights.json` is compressed with `xz -9e` (to
  `weights.json.xz`); its contents are unchanged.

The original licence texts are in `../LICENSES/`, and all third-party sources
are listed in `../NOTICE.md`.
