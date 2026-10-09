//! Local models through llama.cpp (feature `llama-cpp`): running a GGUF model,
//! downloading it, and choosing and loading llama.cpp's compute backends.

mod backends;
mod download;
mod local;

pub use backends::{BackendDevice, backend_devices, load_backends, preferred_gpu};
pub use download::{DownloadProgress, download_model};
pub use local::{LlamaCpp, Sampling};
