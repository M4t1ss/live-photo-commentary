# vlm-describer

Describes screenshots with vision-language models, for live commentary: each
new screenshot is compared with the previous one, with a short history of
earlier comments for continuity. It is an experimental Rust port of the
description code in
[live-photo-commentary](https://github.com/M4t1ss/live-photo-commentary)'s
Python backend.

Models can run in the cloud or locally:

| Backend | Runs | Cargo feature |
|---|---|---|
| `Gemini` | Google's Gemini API | (always available) |
| `OpenAi` | OpenAI's API, or any OpenAI-compatible server (Ollama, LM Studio, llama.cpp's `llama-server`) | (always available) |
| `LlamaCpp` | GGUF models locally, with [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama-cpp` |

The local backend is experimental; [measurements](#measurements-local-models) are below.

## Usage

```rust
use vlm_describer::{Backend, Describer, Gemini};

let backend = Backend::Gemini(Gemini::new("API key", "gemini-3.5-flash"));
let mut describer = Describer::new(backend);
describer.max_history_size = 5;
let first = describer.describe(&previous_screenshot, None).await?;
let next = describer.describe(&screenshot, Some(&previous_screenshot)).await?;
```

With a local model:

```rust
use std::sync::Arc;
use vlm_describer::{Backend, LlamaCpp, Sampling, download_model};

// Downloads the model file and its vision projector from Hugging Face, unless
// they're cached. The optional callback gets the file name, bytes so far and
// size, at most once per percent.
let progress: vlm_describer::DownloadProgress =
    Arc::new(|file, done, total| println!("{file}: {done}/{total}"));
let (model_path, mmproj_path) = download_model(
    "ggml-org/gemma-4-E2B-it-GGUF",
    "gemma-4-E2B-it-Q4_0.gguf",
    "mmproj-gemma-4-E2B-it-Q8_0.gguf",
    Some(progress),
)
.await?;
// The last argument is how many layers to put on the GPU (0 for CPU only, a
// large number for all).
let model = LlamaCpp::new(&model_path, &mmproj_path, 99)?
    .with_sampling(Sampling { top_p: Some(0.95), top_k: Some(64), ..Sampling::default() });
let backend = Backend::LlamaCpp(model);
```

llama.cpp isn't async: describing with `LlamaCpp` blocks the calling thread
until the reply is complete, so call it from a thread of its own (with, for
example, a single-threaded Tokio runtime's `block_on`). Only one `LlamaCpp`
can exist at a time (llama.cpp's backend is global): drop the old one before
loading another.

The crate isn't published; it lives in live-photo-commentary's `crates/`
folder, and the app depends on it by path.

### Examples

```sh
# Cloud models; keys come from GEMINI_API_KEY or OPENAI_API_KEY.
cargo run --example describe -- gemini gemini-3.5-flash shot1.png shot2.png
cargo run --example describe -- openai MODEL --base-url http://localhost:11434/v1 shot1.png

# Local models, downloaded from Hugging Face on first use. Use release
# builds: debug builds are far too slow.
cargo run --release --features llama-cpp,cuda --example local -- \
    llama-cpp ggml-org/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_0.gguf mmproj-gemma-4-E2B-it-Q8_0.gguf \
    shot1.png shot2.png
```

For llama.cpp, set `GPU_LAYERS=99` to run on the GPU; the default, 0, uses
only the CPU.

## Features

| Feature | What it adds | Platforms |
|---|---|---|
| `llama-cpp` | the `LlamaCpp` backend | all |
| `cuda` | NVIDIA GPU support for `llama-cpp` | Windows, Linux |
| `metal` | Apple GPU support for `llama-cpp` | macOS |
| `vulkan` | GPU support for any vendor, for `llama-cpp` | Windows, Linux |

Without a GPU feature, local models run on the CPU, which is slow: about 20 s
to 2 min per screenshot, versus 1–3 s on a mid-range GPU.

`cuda` and `vulkan` have been tested on Windows; `metal` is untested. On an
RTX 3060 Ti, Vulkan was as fast as CUDA with Qwen3-VL-2B and took about 1.35×
as long with Gemma 4 E2B.

## Measurements (local models)

All on an RTX 3060 Ti with 8 GB of video memory, describing screenshots of
1423×799 pixels. Times vary with the length of the reply, so compare averages.

| Model | Device | Per screenshot |
|---|---|---|
| Qwen3-VL-2B Q8_0 | GPU, CUDA | 2.1–3.7 s (average 2.75 s) |
| Qwen3-VL-2B Q8_0 | GPU, Vulkan | 2.0–3.8 s (average 2.9 s) |
| Gemma 4 E2B Q4_0 | GPU, CUDA | 0.9–2.1 s (average 1.3 s) |
| Gemma 4 E2B Q4_0 | GPU, Vulkan | 1.1–2.5 s (average 1.7 s) |
| Qwen3-VL-2B Q8_0 | CPU | 65–125 s (20–37 s at half the image size) |

The app's model catalogue, with Vulkan, four descriptions each and history on.
The first one or two descriptions after a new build are slower while the
driver compiles the Vulkan shaders; it caches them.

| Model | Per screenshot | Notes |
|---|---|---|
| Gemma 4 E2B | 1.4–2.3 s | writes `{tag}`s |
| Gemma 4 E4B | 2.0–4.3 s | tags, sometimes misplaced |
| Gemma 3 4B | 2.3–4.6 s | tags |
| Qwen3-VL 2B | 3.7–5.6 s | no tags |
| Qwen3-VL 4B | 2.5–4.6 s | no tags |
| Qwen3-VL 8B | 4.9–8.1 s | no tags |
| Qwen2.5-VL 3B | 4.6–7.7 s | no tags |
| Qwen2.5-VL 7B | 3.4–6.6 s | no tags |
| InternVL3.5 1B / 2B | 2.2–3.4 s / 1.7–3.8 s | no tags |
| InternVL3.5 4B | 6.4–9 s | no tags |
| InternVL3.5 8B | 4.9–11.5 s | no tags |
| Gemma 3 12B | 15–29 s | partly on the CPU (does not fit in 8 GB) |
| Gemma 4 12B | 12–32 s | partly on the CPU |
| InternVL3.5 14B | 33–60 s | partly on the CPU |

Models that don't write `{tag}`s ignore the prompt's instructions for emotion
tags. Models above 14B (Gemma 4 31B, Qwen3-VL 32B and others) were not tried.

**Video memory.** Gemma 3 4B with Q8_0 weights, an f16 vision projector and an
8192-token context needs about 7 GB. Without care it overflowed the 8 GB card
into system memory on Vulkan: 492 s to load, 114–339 s per screenshot, and it
froze the whole PC. So `LlamaCpp::new`, with `gpu_layers` above 0, asks llama.cpp
(`fit_params`) how many layers fit into the free video memory next to the
context and the image encoder, and runs the rest on the CPU; that case takes
2.3–4.6 s per screenshot. If fitting fails, the model runs on the CPU. The
layers that don't fit are why the 12B and 14B models above are slow.

## Build prerequisites

The cloud backends need nothing beyond Rust.

### Local backend

- **llama.cpp** builds C++ and GPU code from source. It needs a C/C++ compiler, CMake, and libclang (for `bindgen`).
  - **Windows:** Visual Studio Build Tools with the C++ workload, LLVM (set
    `LIBCLANG_PATH`, for example to `C:\Program Files\LLVM\bin`), and CMake.
    With the Visual Studio 2026 Build Tools, CMake 4.2 or newer is needed;
    install it standalone and set `CMAKE`, for example to
    `C:\Program Files\CMake\bin\cmake.exe`.
  - **macOS:** Xcode's command-line tools and CMake.
  - **Debian or Ubuntu:** `build-essential`, `cmake` and `libclang-dev`.

macOS and Linux builds are untested so far.

### `cuda`

Install NVIDIA's [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
On Windows, open a new terminal afterwards so that `CUDA_PATH` and
`CUDA_PATH_V13_4` (or your version) are set; Visual Studio's CUDA build rules
need the latter.

GPU code is compiled for every GPU generation by default, which takes long.
To build only for your GPU, set its compute capability, for example 8.6 for
an RTX 30-series card:

```sh
CMAKE_CUDA_ARCHITECTURES=86
```

At run time, the program needs NVIDIA's cuBLAS libraries (on Windows,
`cublas64_13.dll` and `cublasLt64_13.dll`, about 550 MB), found on `PATH` or
next to the program, and a recent NVIDIA driver. The CUDA runtime itself is
linked statically.

### `vulkan`

Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) and make sure
`VULKAN_SDK` is set. At run time only the GPU driver is needed.

### `metal`

Xcode's command-line tools are enough.

## Licence

Licensed under either of the [MIT licence](LICENSE-MIT) or the
[Apache License, version 2.0](LICENSE-APACHE), at your option.
