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
#[cfg(feature = "llama-cpp")]
mod backends;
mod describer;
#[cfg(feature = "llama-cpp")]
mod download;
mod error;
mod gemini;
#[cfg(feature = "llama-cpp")]
mod local_llamacpp;
mod openai;
mod prompts;

pub use backend::Backend;
#[cfg(feature = "llama-cpp")]
pub use backends::{BackendDevice, backend_devices, load_backends, preferred_gpu};
pub use describer::Describer;
pub use error::{Error, Result};
pub use gemini::Gemini;
#[cfg(feature = "llama-cpp")]
pub use download::{DownloadProgress, download_model};
#[cfg(feature = "llama-cpp")]
pub use local_llamacpp::{LlamaCpp, Sampling};
pub use openai::OpenAi;
pub use prompts::{
    DEFAULT_COMPACT_PROMPT, DEFAULT_FIRST_PROMPT, DEFAULT_HISTORY_PROMPT, DEFAULT_PROMPT,
    DEFAULT_SYSTEM_PROMPT, IMAGE_PLACEHOLDER, Prompts, TAGS_PLACEHOLDER,
};
