//! The error type for everything in this crate.

/// Something that went wrong while describing an image.
///
/// Messages describe only what failed at this level; the underlying cause,
/// if any, is available through [`std::error::Error::source`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The request could not be sent, or its response not read, e.g.
    /// because there is no network.
    #[error("request to {provider} failed")]
    Request {
        provider: &'static str,
        #[source]
        source: reqwest::Error,
    },

    /// The service answered with an error, e.g. a wrong API key or model.
    #[error("{provider} returned HTTP {status}: {body}")]
    Api {
        provider: &'static str,
        status: u16,
        body: String,
    },

    /// The service answered, but without any text, e.g. because the
    /// response was blocked.
    #[error("{provider} returned no text")]
    EmptyResponse { provider: &'static str },

    /// An image could not be encoded as PNG.
    #[error("cannot encode image")]
    Image(#[from] image::ImageError),

    /// A local model could not be loaded or run.
    #[error("local model failed")]
    Local(#[source] Box<dyn std::error::Error + Send + Sync>),

    /// A model repo name isn't of the form `owner/name`.
    #[cfg(feature = "llama-cpp")]
    #[error("invalid Hugging Face repo {0:?}")]
    InvalidRepo(String),

    /// A model file could not be downloaded from Hugging Face, e.g. because
    /// there is no network or no such file.
    #[cfg(feature = "llama-cpp")]
    #[error("cannot download {file} from Hugging Face")]
    Download {
        file: String,
        #[source]
        source: hf_hub::HFError,
    },

    /// The response regex is invalid.
    #[error("invalid response regex")]
    InvalidRegex(#[from] regex::Error),
}

/// `Result` with this crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
