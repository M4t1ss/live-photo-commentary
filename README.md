# Interactive Live Commentary Assistant

A desktop app that comments on what is on your screen, out loud, with an animated avatar. Every few seconds it takes a screenshot, has a vision-language model describe it, speaks the description with the Kokoro text-to-speech model, and lets an avatar lip-sync and act out the emotions the model puts into the text.

It started as a Jupyter notebook for commenting on photos (see "History" below). The app is written in Rust with [Tauri](https://tauri.app); the backend runs inside the app, with no separate server and no Python.

## What it does

- **Describes your screen** with a model of your choice:
  - **locally**, with [llama.cpp](https://github.com/ggml-org/llama.cpp): Gemma 3 and 4, Qwen2.5-VL and Qwen3-VL, InternVL3.5 (the GGUF files are downloaded from Hugging Face on first use), or
  - **in the cloud**: Google Gemini, OpenAI, or any OpenAI-compatible server (Ollama, LM Studio, `llama-server`).
- **Speaks** the description with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), with word-level timings for subtitles and lip-sync, in several voices and languages.
- **Shows an avatar** (a VRM/GLB model with `.vrma` animations) that lip-syncs and plays an animation for each `{emotion}` tag in the text.
- **Skips unchanged screens**: a difference threshold (nine measures, from plain MSE to perceptual hashes) stops it from describing the same thing again.
- **Keeps a short history** of its own comments, so the commentary continues instead of repeating itself.
- **Named promptsets** change the prompts, and with them who the commentator is and what they comment on.

## Installing

Installers for Windows (NSIS), macOS (dmg) and Linux (deb, AppImage) are built by GitHub Actions (`.github/workflows/build.yml`).

- **Avatar model and animations are not included yet** (they are large files). Put your own model in `models/` when building, or upload one in Settings; the avatar needs one.
- **GPU:** local models run on the GPU through Vulkan (Windows, Linux) or Metal (macOS), and on the CPU if there is none (much slower). On an NVIDIA card, Settings can download an optional CUDA backend (about 600 MB, from the GitHub release of the same version).
- **From a version before 0.2.0** (the one with the Python backend): the first start offers to delete the old Python environment, which is several GB. Your settings, promptsets and uploaded models are kept.

## Using it

Open Settings, choose a model provider and model (and enter an API key for a cloud provider), pick a voice, and start. Settings you will probably want to change:

| Setting | What it does |
|---|---|
| Provider, model | Which VLM describes the screen; local models download on first use |
| Voice | The Kokoro voice (a name, or a blend of voices) |
| Pre-screenshot delay | The countdown before each screenshot is taken |
| Difference threshold, measure | How different a screenshot must be from the last one to be described (0 describes every one) |
| History size | How many of its own earlier comments the model sees (0 for none) |
| Promptset | The prompts: system, user, and the greeting, farewell and "lonely" lines |
| Avatar model | Which model the avatar uses; models can be uploaded |

Settings are saved in the app's config folder (`settings.json`); promptsets are YAML files in its data folder.

## Files and folders the app uses

The Windows paths below were checked on an installed copy. The macOS and Linux paths follow Tauri's conventions and the code, but have not been checked on those systems yet.

### Base folders

`<id>` stands for the app's identifier, `io.github.m4t1ss.live-photo-commentary`.

| Name | Windows | macOS | Linux (deb) |
|---|---|---|---|
| **Install folder**, with the executable and the helper | `%LOCALAPPDATA%\Interactive Live Commentary Assistant` | `/Applications/Interactive Live Commentary Assistant.app/Contents/MacOS` | `/usr/bin` |
| **Resources** (libraries, bundled models, licences) | the install folder | `/Applications/Interactive Live Commentary Assistant.app/Contents/Resources` | `/usr/lib/Interactive Live Commentary Assistant` |
| **Config** (`settings.json`) | `%APPDATA%\<id>` | `~/Library/Application Support/<id>` | `~/.config/<id>` |
| **Data** (your files) | `%APPDATA%\<id>` (the same as Config) | `~/Library/Application Support/<id>` (the same) | `~/.local/share/<id>` |
| **Logs** | `%LOCALAPPDATA%\<id>\logs` | `~/Library/Logs/<id>` | `~/.local/share/<id>/logs` |
| **Cache** | `%LOCALAPPDATA%` | `~/Library/Caches` | `~/.cache` (or `$XDG_CACHE_HOME`) |
| **Home** | `C:\Users\<name>` | `/Users/<name>` | `/home/<name>` |

The AppImage has the same layout as the deb, inside the mounted image. The installers are per-user on Windows; the macOS app can be anywhere.

### What is where

| What | Where | Notes |
|---|---|---|
| The program | Install folder: `app.exe` (Windows), `app` (macOS, Linux) | |
| **eSpeak helper** (`kokoro-espeak`) | Install folder (next to the executable) | A separate program that the app starts for pronunciation. `KOKORO_ESPEAK` can point to another one. |
| **eSpeak's data** (`espeak-ng-data`) | Resources | The helper inherits `PIPER_ESPEAKNG_DATA_DIRECTORY`, which the app sets on macOS and Linux (on Windows it is next to the helper). |
| llama.cpp libraries | Resources (the DLLs and `.so`/`.dylib` files), and `backends/` in it for the GPU backend (Vulkan or Metal) | |
| **Bundled avatar model and animations** | Resources, `resources/models/` (`model.glb`, `model.yaml`, `animations/`) | Not in the installers from GitHub yet. Development builds read the repository's `models/` instead. |
| Licences | Resources, `licenses/` | Settings, About, Licences opens it. |
| **Screenshot utility** for WSL (`screenshot.exe`) | Resources, `resources/screenshot.exe`, in Linux packages only | Used only when the Linux app runs inside WSL, where it takes the screenshot on the Windows side. It writes `frames/frame_wsl.png` and `frame_preview.png` in Data (a path under WSL). Native capture needs no file. |
| **Settings** | Config: `settings.json` | Also holds your API keys. API keys are stored as plain text. |
| Window position and size | Config: `.window-state.json` | Written when the app exits normally. |
| **Promptsets** | Data: `prompts/*.yaml` | |
| **Uploaded avatar models** | Data: `user_models/` | |
| **Optional CUDA backend** | Data: `backends/cuda/<app version>/` | `ggml-cuda.dll` (or `libggml-cuda.so`) and NVIDIA's cuBLAS libraries, about 600 MB. A backend for another version is deleted at start. Temporary files while it downloads: `download.zip`, `unpacking/`, and `remove-on-start` after "Remove" while it is loaded. |
| Screenshot files (WSL) | Data: `frames/` | Created always, used only under WSL. |
| Log file | Logs: `Interactive Live Commentary Assistant.log` | Every line is written twice, a known quirk. |
| The window's web data (WebView2 on Windows, WebKit elsewhere) | Windows: `%LOCALAPPDATA%\<id>\EBWebView`. macOS and Linux: managed by the system web view, normally under Caches, Data and `~/Library/WebKit` | Cache and cookies; nothing the app depends on. |
| **Hugging Face hub cache** | `~/.cache/huggingface/hub` (Home), or where `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME` or `XDG_CACHE_HOME` say | Shared with any other Hugging Face tool, Python included. Holds the **Kokoro model and voices** and every **local VLM** you use (GGUF and vision-projector files: several GB for the bigger ones). Nothing in it is removed by uninstalling. |
| Japanese dictionary | Cache: `kokoro-timestamped/unidic-cwj-3_1_1+compact/` | Downloaded (about 24 MB, from the vibrato releases on GitHub) the first time a Japanese voice is used. |
| From versions before 0.2.0 (Python) | Data: `backend/` and `backend-cuda/` | The app offers to delete these once; answering "Keep" leaves an `old-python-env-kept` file in Data. Settings, promptsets and uploaded models were moved out of `backend/` automatically. uv's own Python (`%LOCALAPPDATA%\uv`) is never touched. |

### Downloads and removal

- **Network:** models come from Hugging Face; the optional CUDA backend and the Japanese dictionary come from GitHub. Cloud providers receive screenshots only if you choose one.
- **Uninstalling:** the Windows uninstaller asks whether to remove Config, Data and Logs and the web data (`%APPDATA%\<id>` and `%LOCALAPPDATA%\<id>`), which includes the CUDA backend. It never removes the Hugging Face cache or the dictionary. On macOS and Linux, removing the app leaves all of the user folders above in place.

## Building

You need Rust (1.95 or newer), the [Tauri prerequisites](https://tauri.app/start/prerequisites/), and what llama.cpp and eSpeak NG need to build from source: a C/C++ compiler, CMake, libclang, and the Vulkan SDK for the Vulkan backend. [`crates/vlm-describer/README.md`](crates/vlm-describer/README.md#build-prerequisites) lists them per platform. The repository is a Cargo workspace: the app is in `src-tauri/` and the Rust crates it uses are in `crates/`.

### Run in development

```sh
cargo tauri dev --features devtools,dynamic-backends,vulkan
```

`devtools` adds the DevTools (Ctrl+Shift+I). `dynamic-backends` with `vulkan` (`metal` on macOS) makes llama.cpp load its GPU backend at run time, as the installer does. The first run also builds the `kokoro-espeak` helper. Development builds read the avatar models and animations from `models/` in the repository.

### Tests

```sh
cargo test               # the app and the crates
cargo test -p app --lib  # just the app
```

Tests that need a downloaded model or a GPU are marked `#[ignore]`; the comment on each says how to run it.

### Build an installer

```sh
cargo tauri build --features dynamic-backends,vulkan --config src-tauri/tauri.bundle.conf.json
```

Use `metal` instead of `vulkan` on macOS, and on Linux also add `--config src-tauri/tauri.bundle.linux.conf.json` (it bundles the WSL screenshot helper, which you build with `cargo build --release --manifest-path src-screenshot/Cargo.toml`). `tauri.bundle.conf.json` adds what only installers need: the `kokoro-espeak` helper with `espeak-ng-data` (its `beforeBuildCommand` builds them), the avatar model and animations from `src-tauri/resources/models/`, and the licences. The model and animations are not in the repository: copy `models/model.glb`, `models/model.yaml` and `models/animations/*.vrma` into `src-tauri/resources/models/` first (GitHub's builds put in empty placeholders instead). The installers land in `target/release/bundle/`.

### Other tasks

```sh
cargo xtask package-helper OUT_DIR [--tauri]       # build kokoro-espeak with its espeak-ng-data
cargo xtask package-cuda-backend dist --release    # the downloadable CUDA backend (needs the CUDA Toolkit)
```

The CUDA backend archive is what Settings downloads (from the GitHub release of the same version). `package-cuda-backend` writes the archive and a `.sha256` file; an app built with `CUDA_BACKEND_SHA256` set to that checksum offers the download. The app's version is in `src-tauri/Cargo.toml` and `src-tauri/tauri.conf.json`, and names the release the download comes from.

### On Windows

- Keep Cargo's intermediate files in a short folder (for example `set CARGO_BUILD_BUILD_DIR=C:\b`); llama.cpp's Vulkan build breaks when a path reaches 260 characters, which happens inside this workspace.
- The first Vulkan build often fails once, on a race in llama.cpp's shader generator (`vulkan-shaders-gen`). Run it again.

## Licences

MIT for the app, MIT OR Apache-2.0 for the `kokoro-timestamped` and `vlm-describer` crates, and GPL-3.0-or-later for the separate `kokoro-espeak` helper program (it contains eSpeak NG). The installer includes the licences and the offer of the GPL source (Settings, About, Licences). The details are in [`NOTICE.md`](NOTICE.md).

## History

The first version was a Python notebook (`live_photo_commentary.ipynb`) that commented on photos on screen with a locally run Phi-3.5 vision model or the Gemini API, and Kokoro-82M for speech; a Tauri app with a Python backend followed. Both are in the git history (the notebook and its package, last in commit `8c537af`).

## Example Videos

Recent example with an animated avatar.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ol3w4FaAuZo/0.jpg)](https://www.youtube.com/watch?v=ol3w4FaAuZo)

Initial example without an animated avatar.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/07sU403hMWE/0.jpg)](https://www.youtube.com/watch?v=07sU403hMWE)

## Earlier results: speed and compatibility with the PyTorch models

These were measured with the original Python version, which ran the models through PyTorch and Hugging Face transformers. The Rust app runs GGUF models through llama.cpp instead, so its speeds differ: for example, Gemma 3 4B took 2.3 to 4.6 seconds per screenshot on an RTX 3060 Ti (8 GB) with the Vulkan backend. [`crates/vlm-describer/README.md`](crates/vlm-describer/README.md#measurements-local-models) has those measurements.

The following table shows results in total seconds from testing the description generation on several selected consumer devices. The cases with a `-` denote unsuccesful runs for the specific device and model configuration.

| Model           	| Phi-3.5 	|  Phi-4  	|  Gemma 	|        	| Qwen2.5-VL 	|       	|       	| FastVLM 	|       	|
|-----------------	|--------:	|--------:	|-------:	|-------:	|-----------:	|------:	|------:	|--------:	|------:	|
| Size            	|    4.2B 	|    5.6B 	|     4B 	|    12B 	|         3B 	|    7B 	|  0.5B 	|    1.5B 	|    7B 	|
| RTX 3090        	|    9.77 	|   14.19 	|  25.93 	|  31.37 	|       9.91 	| 10.24 	|  9.06 	|   11.35 	| 12.11 	|
| GTX 1650 Laptop 	|  110.97 	|       - 	| 548.28 	|      - 	|     247.21 	|     - 	| 13.68 	|       - 	|     - 	|
| RTX 3070 Laptop 	|   10.95 	| 1360.62 	|  25.53 	| 421.02 	|      12.41 	| 12.38 	|  6.74 	|    8.37 	| 12.93 	|
| RTX 4070 Laptop 	|   13.58 	| 1270.20 	|  34.56 	| 510.24 	|      19.33 	| 32.87 	| 10.98 	|   16.00 	| 25.78 	|
| RTX 4090 Laptop 	|   10.00 	|   11.03 	|  25.15 	|  32.22 	|      11.11 	| 10.56 	|  8.69 	|   11.64 	| 11.89 	|
| M3 Pro 18GB     	|  103.43 	|       - 	|  38.34 	|      - 	|      20.67 	|     - 	|  7.32 	|   12.87 	|     - 	|
| Average         	|   43.12 	|  664.01 	| 116.30 	| 248.72 	|      53.44 	| 16.51 	|  9.41 	|   12.04 	| 15.68 	|

The different models also tend to generate outputs in different lengths

|            	|         	| 1 Image 	| 2 Images 	|
|------------	| -------:	|:-------:	|:--------:	|
| Phi-3.5    	| 4.2B    	|   363.1 	|    630.9 	|
| Phi-4      	| 5.6B    	|   482.2 	|    659.1 	|
| Gemma      	| 4B      	|   537.3 	|    727.1 	|
|            	| 12B     	|   431.9 	|    474.6 	|
| Qwen2.5-VL 	| 3B      	|   457.6 	|    598.5 	|
|            	| 7B      	|   407.4 	|    639.9 	|
|            	| 0.5B    	|   685.9 	|    779.0 	|
| FastVLM    	| 1.5B    	|   706.1 	|    891.5 	|
|            	| 7B      	|   668.0 	|    788.7 	|
|            	| Average 	|   526.6 	|    687.7 	|

There is also considerable overlap sometimes between the last outputs, especially for smaller models when using history.

|            	|                	| History 0 	|        	| History 1 	|        	| History 5 	|        	|
|------------	|---------------:	|----------:	|-------:	|----------:	|-------:	|----------:	|-------:	|
| Model      	|           Size 	|    tok    	|   chr  	|    tok    	|   chr  	|    tok    	|   chr  	|
| Phi-3.5    	|           4.2B 	|   58.22%  	|  8.77% 	|   90.93%  	| 63.01% 	|   89.47%  	| 43.23% 	|
| Phi-4      	|           5.6B 	|   57.29%  	|  7.08% 	|   96.10%  	| 85.32% 	|   96.10%  	| 54.88% 	|
| Gemma      	|             4B 	|   54.65%  	| 11.28% 	|   76.63%  	| 23.33% 	|   69.97%  	| 23.06% 	|
|            	|            12B 	|   50.15%  	| 10.68% 	|   54.17%  	| 11.75% 	|   45.66%  	| 10.69% 	|
| Qwen2.5-VL 	|             3B 	|   51.54%  	|  7.03% 	|   89.52%  	| 50.77% 	|   79.77%  	| 54.38% 	|
|            	|             7B 	|   48.98%  	|  7.65% 	|   50.65%  	|  7.47% 	|   51.33%  	|  7.70% 	|
| FastVLM    	|           0.5B 	|   56.76%  	| 10.20% 	|   88.54%  	| 42.77% 	|   87.45%  	| 51.12% 	|
|            	|           1.5B 	|   58.87%  	|  9.52% 	|   96.46%  	| 62.42% 	|   98.24%  	| 64.70% 	|
|            	|             7B 	|   57.31%  	|  7.86% 	|   94.90%  	| 26.79% 	|   90.52%  	| 27.40% 	|
| Gemini     	| 2.5-flash-lite 	|   36.50%  	|  7.32% 	|   40.76%  	|  8.39% 	|   41.67%  	|  6.21% 	|
| GPT        	|        4o-mini 	|   44.72%  	| 11.21% 	|   49.67%  	| 10.34% 	|   49.89%  	| 10.74% 	|
