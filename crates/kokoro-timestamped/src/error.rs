//! The error type for everything in this crate.

/// Something that went wrong while loading a synthesizer or synthesizing.
///
/// Messages describe only what failed at this level; the underlying cause,
/// if any, is available through [`std::error::Error::source`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A model or voice file could not be downloaded from Hugging Face,
    /// e.g. because there is no network or no voice by that name.
    #[error("cannot download {file} from Hugging Face")]
    Download {
        file: String,
        #[source]
        source: hf_hub::HFError,
    },

    /// The voices could not be listed: Hugging Face can't be reached, and
    /// nothing has been downloaded yet.
    #[error("cannot list the voices on Hugging Face")]
    ListVoices(#[source] hf_hub::HFError),

    /// A downloaded voice file is not a voice tensor.
    #[error("voice {name:?} is not a valid voice file")]
    InvalidVoice { name: String },

    /// A [`Voice::Blend`](crate::Voice::Blend) without any voices.
    #[error("voice blend must not be empty")]
    EmptyBlend,

    /// The language can't be told from the voice name; pass it explicitly.
    #[error("cannot tell the language of voice {0:?}")]
    UnknownVoiceLanguage(String),

    /// A [`Voice::Array`](crate::Voice::Array) was given without a language.
    #[error("with a voice array, the language must be given")]
    MissingLanguage,

    /// The language isn't supported, e.g. Mandarin (`zh`).
    #[error("unsupported language {0:?}")]
    UnsupportedLanguage(String),

    /// The Japanese dictionary could not be downloaded, unpacked or loaded.
    #[error("cannot load the Japanese dictionary")]
    JapaneseDictionary(#[source] Box<dyn std::error::Error + Send + Sync>),

    /// The eSpeak helper program `kokoro-espeak` was not found next to the
    /// running program, or one folder up. Ship it there, or set
    /// `KOKORO_ESPEAK` to its path.
    #[error("cannot find the eSpeak helper program kokoro-espeak")]
    EspeakHelperNotFound,

    /// The eSpeak helper program could not be started, or stopped working.
    #[error("cannot run the eSpeak helper program")]
    EspeakHelper(#[source] std::io::Error),

    /// eSpeak could not phonemize text. The message is the helper's.
    #[error("eSpeak failed: {0}")]
    Espeak(String),

    /// ONNX Runtime could not load or run the model.
    #[error("ONNX Runtime failed")]
    Onnx(#[from] ort::Error),

    /// A downloaded file could not be read.
    #[error("cannot read a downloaded file")]
    Io(#[from] std::io::Error),
}

/// `Result` with this crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
