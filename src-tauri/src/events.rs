//! The events the pipeline and model lifecycle report, on their way to the
//! frontend. [`Event::to_json`] gives each variant the exact JSON shape that
//! the frontend's message handler (`handleMessage` in `app.js`) reads, and a
//! test pins every shape. The commands send them on a Tauri `Channel`;
//! `pipeline.rs` and `state.rs` only know the [`EventSink`], so they can be
//! unit-tested with the events captured in a `Vec`.

use std::sync::Arc;

use base64::Engine as _;
use image::DynamicImage;

use crate::catalogue::CatalogueEntry;
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

/// One synthesized chunk (sent as the `chunk` event).
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkEvent {
    pub origin: ChunkOrigin,
    pub index: usize,
    pub text: String,
    /// 16-bit PCM mono WAV at `KokoroSynthesizer::SAMPLE_RATE`; `to_json`
    /// base64-encodes this into the `audio_url` data URL.
    pub wav: Arc<[u8]>,
    pub phonemes: Vec<(char, f64)>,
    pub words: Option<Vec<WordTimingEvent>>,
    pub tags: Vec<(String, f64)>,
}

/// Everything the pipeline or model lifecycle can report; `to_json` maps each
/// variant to the JSON shape the frontend reads.
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
    /// Sent while a local model's files or the CUDA backend download.
    #[allow(dead_code)]
    DownloadProgress { file: String, downloaded: u64, total: u64 },
    ModelReady { model: ModelKind },
    ReadyState { ready: bool, vlm: bool, tts: bool, system_messages: bool, running: bool },
    /// `vlm`: the catalogue, in provider order; `tts`: Kokoro voice names
    /// (each becomes `{engine: "kokoro", voice}`).
    Models { vlm: Vec<(String, Vec<CatalogueEntry>)>, tts: Vec<String> },
    PromptsetLoaded { name: String, fields: Promptset, names: Vec<String> },
    Config { data: Config },
}

/// Where the pipeline and model lifecycle send their events. The app's sink
/// forwards them to the frontend's `Channel`; tests capture them in a `Vec`
/// behind a `Mutex`.
pub type EventSink = Arc<dyn Fn(Event) + Send + Sync>;

impl ModelKind {
    fn as_str(self) -> &'static str {
        match self {
            ModelKind::Vlm => "vlm",
            ModelKind::Tts => "tts",
        }
    }
}

/// Rounds `value` to `decimals` decimal places (Python's `round(value, n)`).
fn round(value: f64, decimals: i32) -> f64 {
    let factor = 10f64.powi(decimals);
    (value * factor).round() / factor
}

pub(crate) fn png_data_url(image: &DynamicImage) -> String {
    let mut png = std::io::Cursor::new(Vec::new());
    image.write_to(&mut png, image::ImageFormat::Png).expect("encode PNG");
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(png.into_inner()))
}

fn wav_data_url(wav: &[u8]) -> String {
    format!("data:audio/wav;base64,{}", base64::engine::general_purpose::STANDARD.encode(wav))
}

impl Event {
    /// The exact JSON shape the frontend reads, ready to send on a
    /// `Channel<serde_json::Value>`.
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::json;
        match self {
            Event::Frame { image, push, gen, diff, measure } => {
                let mut value = json!({ "type": "frame", "url": png_data_url(image), "push": push, "gen": gen });
                if let (Some(diff), Some(measure)) = (diff, measure) {
                    value["diff"] = json!(round(*diff, 6));
                    value["measure"] = json!(measure);
                }
                value
            }
            Event::Skipped { diff } => json!({ "type": "skipped", "diff": round(*diff, 6) }),
            Event::Busy { message } => json!({ "type": "busy", "message": message }),
            Event::TakeScreenshot => json!({ "type": "take_screenshot" }),
            Event::Error { message } => json!({ "type": "error", "message": message }),
            Event::Chunk(chunk) => chunk_to_json(chunk),
            Event::TtsDone { gen } => json!({ "type": "tts_done", "gen": gen }),
            Event::SystemTtsDone => json!({ "type": "system_tts_done" }),
            Event::ModelLoading { model } => json!({ "type": "model_loading", "model": model.as_str() }),
            Event::LoadProgress { message } => json!({ "type": "load_progress", "message": message }),
            Event::DownloadProgress { file, downloaded, total } => {
                json!({ "type": "download_progress", "file": file, "downloaded": downloaded, "total": total })
            }
            Event::ModelReady { model } => json!({ "type": "model_ready", "model": model.as_str() }),
            Event::Models { vlm, tts } => {
                let vlm: serde_json::Map<String, serde_json::Value> =
                    vlm.iter().map(|(provider, entries)| (provider.clone(), json!(entries))).collect();
                let tts: Vec<_> = tts.iter().map(|voice| json!({ "engine": "kokoro", "voice": voice })).collect();
                json!({ "type": "models", "vlm": vlm, "tts": tts })
            }
            Event::ReadyState { ready, vlm, tts, system_messages, running } => json!({
                "type": "ready_state",
                "ready": ready,
                "vlm": vlm,
                "tts": tts,
                "system_messages": system_messages,
                "running": running,
            }),
            Event::PromptsetLoaded { name, fields, names } => {
                json!({ "type": "promptset_loaded", "name": name, "fields": fields, "names": names })
            }
            Event::Config { data } => json!({ "type": "config", "data": data }),
        }
    }
}

fn chunk_to_json(chunk: &ChunkEvent) -> serde_json::Value {
    use serde_json::json;
    let mut value = json!({
        "type": "chunk",
        "index": chunk.index,
        "text": chunk.text,
        "audio_url": wav_data_url(&chunk.wav),
        "phonemes": chunk.phonemes.iter().map(|&(ch, t)| json!([ch.to_string(), round(t, 4)])).collect::<Vec<_>>(),
        "tags": chunk.tags.iter().map(|(name, t)| json!({ "name": name, "time": round(*t, 4) })).collect::<Vec<_>>(),
    });
    match chunk.origin {
        ChunkOrigin::Cycle { gen } => value["gen"] = json!(gen),
        ChunkOrigin::System => value["source"] = json!("system"),
    }
    if let Some(words) = &chunk.words {
        let words: Vec<_> = words
            .iter()
            .map(|w| json!({ "s": w.text, "cs": w.char_start, "ce": w.char_end, "ts": round(w.start, 4), "te": round(w.end, 4) }))
            .collect();
        value["words"] = json!(words);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_image() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::new(1, 1))
    }

    #[test]
    fn frame_includes_diff_and_measure_only_when_both_present() {
        let json = Event::Frame { image: Arc::new(tiny_image()), push: true, gen: 3, diff: None, measure: None }.to_json();
        assert_eq!(json["type"], "frame");
        assert_eq!(json["push"], true);
        assert_eq!(json["gen"], 3);
        assert!(json["url"].as_str().unwrap().starts_with("data:image/png;base64,"));
        assert!(json.get("diff").is_none());
        assert!(json.get("measure").is_none());

        let json = Event::Frame {
            image: Arc::new(tiny_image()),
            push: false,
            gen: 4,
            diff: Some(0.123_456_789),
            measure: Some("mse".to_string()),
        }
        .to_json();
        assert_eq!(json["diff"], 0.123457); // rounded to 6 decimals
        assert_eq!(json["measure"], "mse");
    }

    #[test]
    fn skipped_rounds_diff_to_six_decimals() {
        let json = Event::Skipped { diff: 0.123_456_789 }.to_json();
        assert_eq!(json, serde_json::json!({ "type": "skipped", "diff": 0.123457 }));
    }

    #[test]
    fn busy_take_screenshot_and_error_shapes() {
        assert_eq!(
            Event::Busy { message: "VLM busy, cycle skipped".to_string() }.to_json(),
            serde_json::json!({ "type": "busy", "message": "VLM busy, cycle skipped" })
        );
        assert_eq!(Event::TakeScreenshot.to_json(), serde_json::json!({ "type": "take_screenshot" }));
        assert_eq!(
            Event::Error { message: "No VLM configured".to_string() }.to_json(),
            serde_json::json!({ "type": "error", "message": "No VLM configured" })
        );
    }

    #[test]
    fn cycle_chunk_has_gen_not_source_and_rounds_timings() {
        let chunk = ChunkEvent {
            origin: ChunkOrigin::Cycle { gen: 7 },
            index: 0,
            text: "hi".to_string(),
            wav: Arc::from(vec![1, 2, 3]),
            phonemes: vec![('h', 0.123_456), ('i', 0.654_321)],
            words: None,
            tags: vec![("joy".to_string(), 0.1)],
        };
        let json = Event::Chunk(chunk).to_json();
        assert_eq!(json["type"], "chunk");
        assert_eq!(json["gen"], 7);
        assert!(json.get("source").is_none());
        assert_eq!(json["index"], 0);
        assert_eq!(json["text"], "hi");
        assert!(json["audio_url"].as_str().unwrap().starts_with("data:audio/wav;base64,"));
        assert_eq!(json["phonemes"], serde_json::json!([["h", 0.1235], ["i", 0.6543]]));
        assert_eq!(json["tags"], serde_json::json!([{"name": "joy", "time": 0.1}]));
        assert!(json.get("words").is_none());
    }

    #[test]
    fn system_chunk_has_source_not_gen_and_includes_words() {
        let chunk = ChunkEvent {
            origin: ChunkOrigin::System,
            index: 1,
            text: "hello".to_string(),
            wav: Arc::from(Vec::new()),
            phonemes: Vec::new(),
            words: Some(vec![WordTimingEvent { text: "hello".to_string(), char_start: 0, char_end: 5, start: 0.0, end: 0.5 }]),
            tags: Vec::new(),
        };
        let json = Event::Chunk(chunk).to_json();
        assert_eq!(json["source"], "system");
        assert!(json.get("gen").is_none());
        assert_eq!(json["words"], serde_json::json!([{ "s": "hello", "cs": 0, "ce": 5, "ts": 0.0, "te": 0.5 }]));
    }

    #[test]
    fn tts_done_and_system_tts_done_shapes() {
        assert_eq!(Event::TtsDone { gen: 5 }.to_json(), serde_json::json!({ "type": "tts_done", "gen": 5 }));
        assert_eq!(Event::SystemTtsDone.to_json(), serde_json::json!({ "type": "system_tts_done" }));
    }

    #[test]
    fn model_loading_and_model_ready_name_the_model() {
        assert_eq!(
            Event::ModelLoading { model: ModelKind::Vlm }.to_json(),
            serde_json::json!({ "type": "model_loading", "model": "vlm" })
        );
        assert_eq!(
            Event::ModelReady { model: ModelKind::Tts }.to_json(),
            serde_json::json!({ "type": "model_ready", "model": "tts" })
        );
    }

    #[test]
    fn load_progress_and_download_progress_shapes() {
        assert_eq!(
            Event::LoadProgress { message: "Generating messages…".to_string() }.to_json(),
            serde_json::json!({ "type": "load_progress", "message": "Generating messages…" })
        );
        assert_eq!(
            Event::DownloadProgress { file: "model.gguf".to_string(), downloaded: 50, total: 100 }.to_json(),
            serde_json::json!({ "type": "download_progress", "file": "model.gguf", "downloaded": 50, "total": 100 })
        );
    }

    #[test]
    fn ready_state_shape() {
        let json =
            Event::ReadyState { ready: true, vlm: true, tts: true, system_messages: true, running: false }.to_json();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "ready_state", "ready": true, "vlm": true, "tts": true,
                "system_messages": true, "running": false,
            })
        );
    }

    #[test]
    fn models_shape_groups_vlm_entries_by_provider_and_wraps_voices() {
        let entry = CatalogueEntry { name: "gemini-3.5-flash".to_string(), display_name: "Gemini".to_string(), ..Default::default() };
        let json = Event::Models { vlm: vec![("gemini".to_string(), vec![entry])], tts: vec!["af_heart".to_string()] }
            .to_json();
        assert_eq!(json["type"], "models");
        assert_eq!(json["vlm"]["gemini"][0]["name"], "gemini-3.5-flash");
        assert_eq!(json["tts"], serde_json::json!([{ "engine": "kokoro", "voice": "af_heart" }]));
    }

    #[test]
    fn promptset_loaded_and_promptsets_shapes() {
        let fields = Promptset::default();
        let json = Event::PromptsetLoaded {
            name: "default".to_string(),
            fields: fields.clone(),
            names: vec!["default".to_string()],
        }
        .to_json();
        assert_eq!(json["type"], "promptset_loaded");
        assert_eq!(json["name"], "default");
        assert_eq!(json["fields"]["system_prompt"], fields.system_prompt);
        assert_eq!(json["names"], serde_json::json!(["default"]));
    }

    #[test]
    fn config_shape_masks_api_keys_and_has_no_model_dir() {
        let cfg = Config { gemini_api_key: Some("secret".to_string()), ..Config::default() };
        let json = Event::Config { data: cfg.masked() }.to_json();
        assert_eq!(json["type"], "config");
        assert_eq!(json["data"]["gemini_api_key"], "***");
        assert!(json["data"].get("model_dir").is_none(), "model_dir doesn't exist on Config");
    }
}
