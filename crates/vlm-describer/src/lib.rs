//! Describes screenshots with vision-language models, for live commentary.
//!
//! An experimental Rust port of the description code in
//! live-photo-commentary's Python backend. Cloud models (Gemini, and OpenAI
//! or any OpenAI-compatible server) work; local models through llama.cpp
//! (feature `llama-cpp`) are experimental.
//!
//! ```no_run
//! use vlm_describer::{Backend, Describer, Gemini};
//!
//! # async fn run(screenshot: image::DynamicImage, previous: image::DynamicImage)
//! # -> vlm_describer::Result<()> {
//! let backend = Backend::Gemini(Gemini::new("API key", "gemini-3.5-flash"));
//! let mut describer = Describer::new(backend);
//! describer.max_history_size = 5;
//! let first = describer.describe(&previous, None).await?;
//! let next = describer.describe(&screenshot, Some(&previous)).await?;
//! # Ok(())
//! # }
//! ```

mod backend;
mod describer;
mod error;
mod gemini;
#[cfg(feature = "llama-cpp")]
mod llama_cpp;
mod openai;
mod prompts;

pub use backend::Backend;
pub use describer::Describer;
pub use error::{Error, Result};
pub use gemini::Gemini;
#[cfg(feature = "llama-cpp")]
pub use llama_cpp::{
    BackendDevice, DownloadProgress, LlamaCpp, Sampling, backend_devices, download_model, load_backends,
    preferred_gpu,
};
pub use openai::OpenAi;
pub use prompts::{
    DEFAULT_COMPACT_PROMPT, DEFAULT_FIRST_PROMPT, DEFAULT_HISTORY_PROMPT, DEFAULT_PROMPT,
    DEFAULT_SYSTEM_PROMPT, IMAGE_PLACEHOLDER, Prompts, TAGS_PLACEHOLDER,
};
