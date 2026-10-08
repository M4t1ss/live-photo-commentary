# Notices

kokoro-timestamped is licensed under either of the MIT licence
(`LICENSE-MIT`) or the Apache License, version 2.0 (`LICENSE-APACHE`), at
your option.

Phonemization uses [eSpeak NG](https://github.com/espeak-ng/espeak-ng)
(GPL-3.0-or-later), which runs in the separate helper program
`kokoro-espeak` in this workspace. The helper is GPL-3.0-or-later and has its
own notices in `../kokoro-espeak/NOTICE.md`; this crate only starts it and
exchanges text with it, so it is not covered by the GPL.

Parts of this crate are ported from, or contain data from, the projects below.
Their licence texts are in `LICENSES/`. Each ported Rust module says at the top
what it was ported from and how it differs.

## Ported code

| Rust | Ported from | Licence |
|---|---|---|
| `src/japanese.rs` (including the kana table, generated from misaki) | `misaki/cutlet.py` from [misaki](https://github.com/hexgrad/misaki) by hexgrad, itself adapted from [cutlet](https://github.com/polm/cutlet), Copyright (c) 2020 Paul O'Leary McCann | Apache-2.0; cutlet: MIT (`LICENSES/cutlet-MIT.txt`) |
| `src/num2kana.rs` | `misaki/num2kana.py` from misaki, copied from [Convert-Numbers-to-Japanese](https://github.com/Greatdane/Convert-Numbers-to-Japanese), Copyright (c) 2018 David Wilson | Apache-2.0; original: MIT (`LICENSES/Convert-Numbers-to-Japanese-MIT.txt`) |
| `src/english.rs`: heteronym lookup | `Lexicon.lookup` and `Lexicon.get_parent_tag` in `misaki/en.py` | Apache-2.0 |
| `src/english/pos_tagger.rs` | `nltk/tag/perceptron.py` from [NLTK](https://github.com/nltk/nltk), Copyright (C) NLTK Project, itself a port of [textblob-aptagger](https://github.com/sloria/textblob-aptagger), Copyright 2013 Matthew Honnibal | Apache-2.0; textblob-aptagger: MIT (`LICENSES/textblob-aptagger-MIT.txt`) |
| `VOCAB` in `src/kokoro.rs` | `config.json` of [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad | Apache-2.0 |

The Apache-2.0 licence text is in `LICENSE-APACHE`.

The pipeline as a whole (`src/kokoro.rs`, `src/text_segmentation.rs`, `src/g2p.rs`)
is ported from the Python backend of
[live-photo-commentary](https://github.com/M4t1ss/live-photo-commentary)
(MIT), from code written by the same author as this crate.

## Data

See `data/README.md` for the sources and licences of everything in `data/`:

- the heteronym tables and Japanese word list, from misaki (Apache-2.0)
- the part-of-speech tagger weights, from NLTK's averaged perceptron model, as converted to JSON by [misaki-rs](https://github.com/MicheleYin/misaki-rs) (MIT, `LICENSES/misaki-rs-MIT.txt`)

## Downloaded at runtime

These are not part of this crate, but are downloaded when it runs:

- **Model and voices:** the Kokoro-82M ONNX model and voices from [onnx-community/Kokoro-82M-v1.0-ONNX-timestamped](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped) (Apache-2.0).
- **Japanese dictionary:** [UniDic](https://clrd.ninjal.ac.jp/unidic/) 3.1.1 (GPL-2.0, LGPL-2.1 or BSD-3-Clause), via [vibrato](https://github.com/daac-tools/vibrato)'s prebuilt dictionaries.
