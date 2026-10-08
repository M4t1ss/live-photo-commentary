# Notices

kokoro-espeak is free software, licensed under the GNU General Public
License, version 3 or later (see `LICENSE`).

It is GPL because it statically links [eSpeak NG](https://github.com/espeak-ng/espeak-ng)
(GPL-3.0-or-later), and because its punctuation handling is modelled on
[phonemizer](https://github.com/bootphon/phonemizer) (GPL-3.0-or-later).

It is a separate program so that kokoro-timestamped, which starts it and
exchanges text with it over stdin and stdout, can be permissively licensed.
Anyone distributing this program (for example in an installer) must also
offer its complete source code, including eSpeak NG's, under the GPL, and
ship eSpeak NG's `espeak-ng-data` directory, which is GPL too.

## Ported code

| Rust | Ported from | Licence |
|---|---|---|
| `src/espeak.rs`: the `E2M` table and eSpeak conversion | `EspeakG2P` in `misaki/espeak.py` from [misaki](https://github.com/hexgrad/misaki) by hexgrad | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |
| `src/espeak.rs`: `MARKS`, `split_marks`, and putting punctuation back in `phonemize` | `Punctuation` in `phonemizer/punctuation.py` from phonemizer | GPL-3.0-or-later |
