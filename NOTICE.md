# Notices

This repository contains code under different licences, by directory:

| Directory | Licence |
|---|---|
| everything not listed below (the app: `src-tauri/`, `frontend/`, …) | MIT (`LICENSE`) |
| `crates/kokoro-timestamped/` | MIT OR Apache-2.0 (its `LICENSE-MIT`, `LICENSE-APACHE`) |
| `crates/vlm-describer/` | MIT OR Apache-2.0 (its `LICENSE-MIT`, `LICENSE-APACHE`) |
| `xtask/` | MIT OR Apache-2.0 |
| `crates/kokoro-espeak/` | GPL-3.0-or-later (its `LICENSE`) |

`kokoro-espeak` is GPL because it statically links eSpeak NG. It is a
separate program: kokoro-timestamped starts it and exchanges plain text with
it over stdin and stdout, so the library and the app that uses it are not
covered by the GPL. Distributing `kokoro-espeak` (for example in an
installer) requires offering its complete source, including eSpeak NG's; see
"Source code of the GPL parts" below.

Third-party code and data ported into or bundled with the app and the crates
are listed in their own notices, with the original licence texts in their
`LICENSES/` folders:

- `src-tauri/NOTICE.md`
- `crates/kokoro-timestamped/NOTICE.md` (and `data/README.md` there)
- `crates/kokoro-espeak/NOTICE.md`

## Source code of the GPL parts

The installer contains `kokoro-espeak` (GPL-3.0-or-later; its licence text is
`crates/kokoro-espeak/LICENSE`, installed as `licenses/kokoro-espeak/LICENSE`)
and eSpeak NG's `espeak-ng-data` folder. Their complete corresponding source
code is available, for at least three years after the version was released:

- `kokoro-espeak`: the `crates/kokoro-espeak` directory of
  https://github.com/M4t1ss/live-photo-commentary, at the tag `v<version>` of the
  installed version (shown in Settings, UI tab, About).
- eSpeak NG as built into it: the `espeak-rs-sys` crate (version 0.2.0, on
  crates.io) contains the sources that were compiled; upstream is
  https://github.com/espeak-ng/espeak-ng.
- If you cannot get the source from there, open an issue at
  https://github.com/M4t1ss/live-photo-commentary/issues and it will be sent to you.

In the installed app, Settings > UI > About > Licences opens the folder with
this file, the licences of the Rust libraries the app is built from
(`app/LICENSES/THIRD-PARTY-LICENSES.html`) and the notices of the individual parts.

## NVIDIA components (optional CUDA download)

The installer contains no NVIDIA code. The CUDA backend, which Settings can
download, is a zip file containing:

- `ggml-cuda.dll` / `libggml-cuda.so`: llama.cpp's CUDA backend (MIT), built
  from the sources of the `llama-cpp-sys-2` crate;
- NVIDIA's cuBLAS and cuBLASLt libraries (`cublas64_13.dll`,
  `cublasLt64_13.dll`, or `libcublas.so.13`, `libcublasLt.so.13`), which are
  proprietary. They are redistributed under the CUDA Toolkit licence agreement
  (the "EULA", Attachment A lists them as distributable). That agreement is in the
  zip as `NVIDIA-CUDA-EULA.txt`, and ends up next to the libraries in
  `<app data>/backends/cuda/<version>/`. It also reproduces the notices of the
  third-party code inside cuBLAS. The libraries are used only by this app and are
  not covered by this repository's licences.
