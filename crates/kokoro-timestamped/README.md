# kokoro-timestamped

Text-to-speech with
[Kokoro-82M-v1.0-ONNX-timestamped](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped),
an ONNX export of [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) that
also outputs phoneme durations, in Rust, with timings. Besides audio, synthesis returns when each phoneme, each
word and each `{tag}` in the text is spoken, for lip-sync, subtitles and
things like avatar emotions.

It is a Rust port of the speech synthesis in
[live-photo-commentary](https://github.com/M4t1ss/live-photo-commentary)'s
Python backend.

- **Languages:** American and British English, Spanish, French, Hindi,
  Italian, Brazilian Portuguese and Japanese. The language is taken from the
  voice name (`af_heart` is American English).
- **Runs on the CPU** through ONNX Runtime, which is fast enough for Kokoro.
- **Downloads what it needs on first use:** the model and voices from Hugging
  Face, and the Japanese dictionary on first use of a Japanese voice.

## Usage

```rust
use kokoro_timestamped::{KokoroSynthesizer, Voice};

let mut synthesizer =
    KokoroSynthesizer::new(Voice::Name("af_heart".into()), "model", None, 1.0).await?;
for chunk in synthesizer.synthesize("{happy} Hello, world!")? {
    let chunk = chunk?;
    // Play chunk.audio (mono f32 at KokoroSynthesizer::SAMPLE_RATE),
    // using chunk.phonemes, chunk.words and chunk.tags for timing.
}

// All voice names, e.g. for a settings menu (cached ones when offline).
let voices = KokoroSynthesizer::list_voices().await?;
```

The crate isn't published; it lives in live-photo-commentary's `crates/`
folder, and the app depends on it by path.

To try it:

```sh
cargo build
cargo run --example say -- --voice af_heart "Hello, world!"   # writes output.wav
```

## The eSpeak helper program

Languages other than Japanese are phonemized with
[eSpeak NG](https://github.com/espeak-ng/espeak-ng). eSpeak NG is GPL, so it
runs in a separate helper program, `kokoro-espeak` (in this workspace), and
the library talks to it over stdin and stdout. This keeps the library itself
permissively licensed (see [Licence](#licence)).

`cargo build` builds both. `cargo run --example say` builds only the library,
so after changing the helper, run `cargo build` first.

## Build prerequisites

Only the helper needs these, because it builds eSpeak NG from its C sources
with CMake and generates bindings with `bindgen`. The library itself has no
special requirements.

### Windows

- **Visual Studio Build Tools** with the C++ workload.
- **LLVM** for libclang. Set `LIBCLANG_PATH`, for example to
  `C:\Program Files\LLVM\bin`.
- **CMake.** With the Visual Studio 2026 Build Tools, CMake 4.2 or newer is
  needed, because older versions don't know the `Visual Studio 18 2026`
  generator; the CMake bundled with Visual Studio 2022 is too old. Install
  standalone CMake and set `CMAKE`, for example to
  `C:\Program Files\CMake\bin\cmake.exe`.
- **`/utf-8`** is already set in the workspace's `.cargo/config.toml`, so there's nothing to
  do. It's needed because eSpeak NG's sources are UTF-8, and without it MSVC
  reads them in the system's code page. On Windows set to a Japanese,
  Chinese or Korean locale (code pages 932, 936, 949, 950), the build then
  fails. A program that builds eSpeak NG outside this
  workspace needs the same setting in its own `.cargo/config.toml`.

eSpeak NG's CMake settings are cached after the first build. After changing
any of the above, run `cargo clean -p espeak-rs-sys`.

### macOS and Linux

Not tested yet. You need a C/C++ compiler, CMake, and libclang:

- **macOS:** Xcode's command-line tools (`xcode-select --install`) and CMake
  (for example `brew install cmake`).
- **Debian or Ubuntu:** `build-essential`, `cmake` and `libclang-dev`.

## Shipping a program that uses it

Depending on this crate does **not** build the helper: Cargo only builds a
dependency's library, never its programs or other workspace members. Your
build has to do it, with this workspace's `package-helper` task:

```sh
cargo run --release --manifest-path path/to/live-photo-commentary/Cargo.toml -p xtask -- \
    package-helper OUT_DIR [--target TRIPLE] [--tauri]
```

(or `cargo xtask package-helper ...` from inside this workspace). It builds
`kokoro-espeak` in release mode and copies it, with eSpeak's `espeak-ng-data`
folder, into `OUT_DIR`. With `--tauri`, the helper is named
`kokoro-espeak-<target-triple>`, as Tauri sidecars require. In a Tauri app,
run it from `beforeDevCommand` and `beforeBuildCommand`. With a git
dependency, `cargo metadata` tells you where Cargo checked the crate out.

A program built with this crate needs these at run time:

- **The `kokoro-espeak` helper.** It's found next to the running program (or
  one folder up), or through `KOKORO_ESPEAK`, set to its path. In a Tauri
  app, bundle it as a sidecar (`bundle.externalBin`).
- **eSpeak's `espeak-ng-data` folder**, which `package-helper` copies. In
  development the helper finds it inside `target/`. When shipping, put it
  next to `kokoro-espeak`, or set `PIPER_ESPEAKNG_DATA_DIRECTORY` to the
  folder that contains it. On macOS and Linux, Tauri puts bundled files
  elsewhere (`Contents/Resources`, `/usr/lib/<app>/`), so set the variable
  there; the helper inherits it.
- **On Windows, a delay-loaded `DirectML.dll`.** ONNX Runtime's prebuilt
  Windows binaries always link DirectML, which this crate never uses.
  Delay-loading it means the DLL never needs to exist. This crate's
  `build.rs` does it for its own examples and tests, but build-script link
  arguments only apply to their own package, so your program needs this in
  its own `build.rs`:

  ```rust
  println!("cargo:rustc-link-arg=/DELAYLOAD:DirectML.dll");
  println!("cargo:rustc-link-arg=delayimp.lib");
  ```

  (only when building for Windows with MSVC).
- **On Windows, the Visual C++ runtime**, usually already installed.

Distributing `kokoro-espeak` means meeting the GPL's terms for it: offer its
complete source, including eSpeak NG's.

## Licence

The library is licensed under either of the [MIT licence](LICENSE-MIT) or the
[Apache License, version 2.0](LICENSE-APACHE), at your option.

The `kokoro-espeak` helper is licensed under the
[GNU General Public License, version 3 or later](../kokoro-espeak/LICENSE),
because it links eSpeak NG.

Ported code and data come from misaki, cutlet, NLTK and others; see
[NOTICE.md](NOTICE.md) and [kokoro-espeak/NOTICE.md](../kokoro-espeak/NOTICE.md).
