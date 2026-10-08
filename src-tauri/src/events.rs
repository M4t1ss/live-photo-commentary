//! The events the pipeline and model lifecycle report, on their way to the
//! frontend (RUSTIFICATION.md's Appendix A).
//!
//! Phase 4 gives these the exact JSON shape, wires them into a Tauri
//! `Channel`, and pins each variant's shape with a serialization test; for
//! now this is a plain Rust enum, used by `pipeline.rs` and `state.rs` so
//! they can be developed and unit-tested before anything is wired in.

use std::sync::Arc;

use image::DynamicImage;

use crate::config::Config;
use crate::promptsets::Promptset;

/// Which model a loading/ready message is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Vlm,
    Tts,
}

/// Whether a synthesized chunk belongs to a cycle (has a generation) or a
/// system message (greeting/farewell/lonely, or ad-hoc `/synthesize`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkOrigin {
    Cycle { gen: u64 },
    System,
}

/// A word's timing, owned and `Clone` (unlike [`kokoro_timestamped::WordTiming`])
/// so pre-generated system messages can be replayed more than once.
#[derive(Clone, Debug, PartialEq)]
pub struct WordTimingEvent {
    pub text: String,
    pub char_start: usize,
    pub char_end: usize,
    pub start: f64,
    pub end: f64,
}

impl From<kokoro_timestamped::WordTiming> for WordTimingEvent {
    fn from(w: kokoro_timestamped::WordTiming) -> Self {
        Self { text: w.text, char_start: w.char_start, char_end: w.char_end, start: w.start, end: w.end }
    }
}

/// One synthesized chunk (`chunk` in Appendix A).
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkEvent {
    pub origin: ChunkOrigin,
    pub index: usize,
    pub text: String,
    /// 16-bit PCM mono WAV at `KokoroSynthesizer::SAMPLE_RATE`; phase 4
    /// base64-encodes this into the `audio_url` data URL.
    pub wav: Arc<[u8]>,
    pub phonemes: Vec<(char, f64)>,
    pub words: Option<Vec<WordTimingEvent>>,
    pub tags: Vec<(String, f64)>,
}

/// Everything the pipeline or model lifecycle can report. Phase 4 maps each
/// variant to the JSON shape in Appendix A.
#[derive(Debug)]
pub enum Event {
    Frame { image: Arc<DynamicImage>, push: bool, gen: u64, diff: Option<f64>, measure: Option<String> },
    Skipped { diff: f64 },
    Busy { message: String },
    TakeScreenshot,
    Error { message: String },
    Chunk(ChunkEvent),
    TtsDone { gen: u64 },
    SystemTtsDone,
    ModelLoading { model: ModelKind },
    LoadProgress { message: String },
    DownloadProgress { file: String, downloaded: u64, total: u64 },
    ModelReady { model: ModelKind },
    ReadyState { ready: bool, vlm: bool, tts: bool, system_messages: bool, running: bool },
    PromptsetLoaded { name: String, fields: Promptset, names: Vec<String> },
    Promptsets { names: Vec<String> },
    Config { data: Config },
}

/// Where the pipeline and model lifecycle send their events. A `Channel`
/// wraps this in phase 4; tests capture events in a `Vec` behind a `Mutex`.
pub type EventSink = Arc<dyn Fn(Event) + Send + Sync>;
