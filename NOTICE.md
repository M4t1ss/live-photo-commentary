# Notices

This repository contains code under different licences, by directory:

| Directory | Licence |
|---|---|
| everything not listed below (the app: `src-tauri/`, `frontend/`, `backend/`, …) | MIT (`LICENSE`) |
| `crates/kokoro-timestamped/` | MIT OR Apache-2.0 (its `LICENSE-MIT`, `LICENSE-APACHE`) |
| `crates/vlm-describer/` | MIT OR Apache-2.0 (its `LICENSE-MIT`, `LICENSE-APACHE`) |
| `xtask/` | MIT OR Apache-2.0 |
| `crates/kokoro-espeak/` | GPL-3.0-or-later (its `LICENSE`) |

`kokoro-espeak` is GPL because it statically links eSpeak NG. It is a
separate program: kokoro-timestamped starts it and exchanges plain text with
it over stdin and stdout, so the library and the app that uses it are not
covered by the GPL. Distributing `kokoro-espeak` (for example in an
installer) requires offering its complete source, including eSpeak NG's.

Third-party code and data ported into or bundled with the app and the crates
are listed in their own notices, with the original licence texts in their
`LICENSES/` folders:

- `src-tauri/NOTICE.md`
- `crates/kokoro-timestamped/NOTICE.md` (and `data/README.md` there)
- `crates/kokoro-espeak/NOTICE.md`
