# Notices for whisper-helper

`whisper-helper` is MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`). It
contains, linked in, these third-party works, all under licences that allow
redistribution with a notice:

- **whisper.cpp** by Georgi Gerganov and contributors, MIT licence
  (https://github.com/ggml-org/whisper.cpp), built by the `whisper-rs-sys`
  crate from the sources it contains. This includes **ggml**, also MIT,
  copyright the ggml authors.
- **whisper-rs** (Rust bindings for whisper.cpp), published under the
  Unlicense (public domain dedication).
- With the `vulkan` feature, the Vulkan backend of ggml (MIT) and the Khronos
  Vulkan headers (Apache-2.0 / MIT); the program loads the system's Vulkan
  library at run time.

The speech recognition models the app downloads (`ggml-*-q5_*.bin` from
https://huggingface.co/ggerganov/whisper.cpp) are converted copies of OpenAI's
Whisper models, which OpenAI released under the MIT licence.

The licences of the Rust libraries the helper is built from are in
`app/LICENSES/THIRD-PARTY-LICENSES.html`, next to this folder in the installed
app's licence folder.

MIT licence text (whisper.cpp, ggml, Whisper models):

    MIT License

    Copyright (c) 2023-2026 The ggml authors

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
