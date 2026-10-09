//! Text-to-speech with [Kokoro-82M-v1.0-ONNX-timestamped], an ONNX export of
//! [Kokoro-82M] that also outputs how long each phoneme lasts, ported from the
//! Python backend of live-photo-commentary.
//!
//! Besides audio, synthesis returns timings for each phoneme (for lip-sync),
//! each word (for subtitles), and each `{tag}` in the text (for things like
//! avatar emotions). Supported languages are American and British English,
//! Spanish, French, Hindi, Italian, Brazilian Portuguese and Japanese; the
//! language is taken from the voice name.
//!
//! The model and voices are downloaded from Hugging Face on first use, and the
//! Japanese dictionary on first use of a Japanese voice.
//!
//! Languages other than Japanese are phonemized with eSpeak NG, which runs in
//! the separate helper program `kokoro-espeak` (GPL, from this crate's
//! workspace). Ship it next to your program, or set `KOKORO_ESPEAK` to its
//! path; it needs eSpeak's `espeak-ng-data` directory next to it.
//!
//! ```no_run
//! use kokoro_timestamped::{KokoroSynthesizer, Voice};
//!
//! # async fn run() -> kokoro_timestamped::Result<()> {
//! let mut synthesizer =
//!     KokoroSynthesizer::new(Voice::Name("af_heart".into()), "model", None, 1.0).await?;
//! for chunk in synthesizer.synthesize("{happy} Hello, world!")? {
//!     let chunk = chunk?;
//!     // play chunk.audio (mono f32 at KokoroSynthesizer::SAMPLE_RATE)
//!     // while using chunk.phonemes, chunk.words and chunk.tags
//! }
//! # Ok(())
//! # }
//! ```
//!
//! [Kokoro-82M-v1.0-ONNX-timestamped]: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped
//! [Kokoro-82M]: https://huggingface.co/hexgrad/Kokoro-82M

mod embedded;
mod english;
mod error;
mod espeak;
mod g2p;
mod japanese;
mod kokoro;
mod num2kana;
mod text_segmentation;

pub use error::{Error, Result};
pub use kokoro::{Chunk, KokoroSynthesizer, Voice, VoiceBlendPart, WordTiming};
