//! Speech recognition: whisper.cpp through whisper-rs, run on a thread of its
//! own because transcribing blocks for a second or more.

use std::path::Path;
use std::sync::mpsc;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// A segment whisper thinks is more likely silence than speech is dropped
/// when its probability of that is above this. Whisper makes up text such as
/// "Thank you." from noise.
const NO_SPEECH_MAX: f32 = 0.6;
/// Anything shorter than this (16 kHz samples) is not worth transcribing.
const MIN_SAMPLES: usize = 16_000 / 2;

/// What the voice input needs from a speech recognizer. Implemented for
/// [`Whisper`]; tests implement it for fakes.
pub trait AsrOps: Send {
    /// What was said in `samples` (16 kHz mono). `language` is a code such as
    /// `"en"`; `None` detects it. Empty if there was no speech.
    fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<String, String>;
}

pub struct Whisper {
    context: WhisperContext,
}

impl Whisper {
    /// Loads the model file at `path`. Runs on the CPU: the GPU is llama.cpp's,
    /// and a few seconds of speech don't need more.
    pub fn load(path: &Path) -> Result<Self, String> {
        static LOGGING: std::sync::Once = std::sync::Once::new();
        LOGGING.call_once(whisper_rs::install_logging_hooks);

        let path = path.to_str().ok_or_else(|| format!("{} is not valid UTF-8", path.display()))?;
        let mut params = WhisperContextParameters::default();
        params.use_gpu(false);
        let context = WhisperContext::new_with_params(path, params).map_err(|e| format!("cannot load {path}: {e}"))?;
        Ok(Self { context })
    }
}

impl AsrOps for Whisper {
    fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<String, String> {
        if samples.len() < MIN_SAMPLES {
            return Ok(String::new());
        }
        let mut state = self.context.create_state().map_err(|e| e.to_string())?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(language.unwrap_or("auto")));
        params.set_n_threads(std::thread::available_parallelism().map_or(4, |n| n.get().min(4)) as i32);
        params.set_translate(false);
        params.set_no_context(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        state.full(params, samples).map_err(|e| format!("transcription failed: {e}"))?;

        let mut text = String::new();
        for segment in state.as_iter() {
            if segment.no_speech_probability() > NO_SPEECH_MAX {
                continue;
            }
            text.push_str(&segment.to_str_lossy().map_err(|e| e.to_string())?);
        }
        Ok(clean_transcript(&text))
    }
}

/// Takes the annotations whisper adds for sounds, such as `[BLANK_AUDIO]`,
/// `(music)` and ♪, out of `text`, and tidies the spaces. A bracket with no
/// partner is kept as it is.
pub fn clean_transcript(text: &str) -> String {
    let without = remove_between(&remove_between(text, '[', ']'), '(', ')').replace('♪', " ");
    without.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `text` without each `open`...`close` pair and what is between them.
fn remove_between(text: &str, open: char, close: char) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        match rest[start + open.len_utf8()..].find(close) {
            Some(length) => {
                out.push_str(&rest[..start]);
                out.push(' ');
                rest = &rest[start + open.len_utf8() + length + close.len_utf8()..];
            }
            None => break,
        }
    }
    out.push_str(rest);
    out
}

enum AsrJob {
    SetModel(Option<Box<dyn AsrOps>>, mpsc::Sender<()>),
    Transcribe { samples: Vec<f32>, language: Option<String>, done: Box<dyn FnOnce(Result<String, String>) + Send> },
}

/// A thread that owns the recognizer and runs the transcriptions one at a
/// time, in the order they were asked for.
pub struct AsrWorker {
    tx: mpsc::Sender<AsrJob>,
}

impl AsrWorker {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<AsrJob>();
        std::thread::spawn(move || {
            let mut model: Option<Box<dyn AsrOps>> = None;
            while let Ok(job) = rx.recv() {
                match job {
                    AsrJob::SetModel(new, ack) => {
                        model = new;
                        let _ = ack.send(());
                    }
                    AsrJob::Transcribe { samples, language, done } => {
                        let result = match model.as_mut() {
                            Some(model) => model.transcribe(&samples, language.as_deref()),
                            None => Err("No speech recognition model is loaded".to_string()),
                        };
                        done(result);
                    }
                }
            }
        });
        Self { tx }
    }

    /// Replaces the recognizer (`None` frees it). Waits until any
    /// transcription in progress has finished, so call it from a thread that
    /// may block.
    pub fn set_model(&self, model: Option<Box<dyn AsrOps>>) {
        let (ack, rx) = mpsc::channel();
        if self.tx.send(AsrJob::SetModel(model, ack)).is_ok() {
            let _ = rx.recv();
        }
    }

    /// Transcribes `samples` (16 kHz mono) and calls `done` on the worker
    /// thread with the text, or the error.
    pub fn transcribe(
        &self,
        samples: Vec<f32>,
        language: Option<String>,
        done: impl FnOnce(Result<String, String>) + Send + 'static,
    ) {
        if let Err(mpsc::SendError(AsrJob::Transcribe { done, .. })) =
            self.tx.send(AsrJob::Transcribe { samples, language, done: Box::new(done) })
        {
            done(Err("The speech recognition thread has stopped".to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn annotations_for_sounds_are_removed() {
        assert_eq!(clean_transcript(" [BLANK_AUDIO] "), "");
        assert_eq!(clean_transcript("(music)"), "");
        assert_eq!(clean_transcript("♪ ♪"), "");
        assert_eq!(clean_transcript("Hello [MUSIC] there (laughs) friend."), "Hello there friend.");
        assert_eq!(clean_transcript("  what   is\nthat? "), "what is that?");
    }

    #[test]
    fn a_bracket_without_a_partner_is_kept() {
        assert_eq!(clean_transcript("a [ b"), "a [ b");
        assert_eq!(clean_transcript("x ) y ( z"), "x ) y ( z");
        assert_eq!(clean_transcript("a [b] c [ d"), "a c [ d");
    }

    #[test]
    fn real_words_are_left_alone() {
        assert_eq!(clean_transcript("What's on my screen, please?"), "What's on my screen, please?");
        assert_eq!(clean_transcript("日本語のテスト"), "日本語のテスト");
    }

    /// Says what it was given, and that it was.
    struct Echo;

    impl AsrOps for Echo {
        fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<String, String> {
            Ok(format!("{} samples in {}", samples.len(), language.unwrap_or("auto")))
        }
    }

    fn transcribe(worker: &AsrWorker, samples: usize, language: Option<&str>) -> Result<String, String> {
        let (tx, rx) = mpsc::channel();
        worker.transcribe(vec![0.0; samples], language.map(str::to_string), move |result| {
            let _ = tx.send(result);
        });
        rx.recv_timeout(Duration::from_secs(5)).expect("the worker should answer")
    }

    #[test]
    fn the_worker_transcribes_with_the_model_it_was_given() {
        let worker = AsrWorker::new();
        worker.set_model(Some(Box::new(Echo)));
        assert_eq!(transcribe(&worker, 100, Some("ja")).unwrap(), "100 samples in ja");
        assert_eq!(transcribe(&worker, 7, None).unwrap(), "7 samples in auto");
    }

    #[test]
    fn without_a_model_it_is_an_error_and_a_removed_model_is_gone() {
        let worker = AsrWorker::new();
        assert!(transcribe(&worker, 100, None).is_err());
        worker.set_model(Some(Box::new(Echo)));
        assert!(transcribe(&worker, 100, None).is_ok());
        worker.set_model(None);
        assert!(transcribe(&worker, 100, None).unwrap_err().contains("No speech recognition model"));
    }

    #[test]
    fn requests_are_answered_in_order() {
        let worker = AsrWorker::new();
        worker.set_model(Some(Box::new(Echo)));
        let (tx, rx) = mpsc::channel();
        for n in 1..=5usize {
            let tx = tx.clone();
            worker.transcribe(vec![0.0; n], None, move |result| {
                let _ = tx.send(result.unwrap());
            });
        }
        let answers: Vec<String> = (0..5).map(|_| rx.recv_timeout(Duration::from_secs(5)).unwrap()).collect();
        assert_eq!(answers, (1..=5).map(|n| format!("{n} samples in auto")).collect::<Vec<_>>());
    }

    /// Loads a real whisper model and transcribes a 16 kHz mono 16-bit WAV
    /// with it, printing the text and the time. Needs both files; the sample
    /// is whisper.cpp's `samples/jfk.wav`, and a model from
    /// `ggerganov/whisper.cpp` (this app's cache has them once it has used
    /// voice input). Run on request:
    /// `ASR_MODEL=ggml-base-q5_1.bin ASR_WAV=jfk.wav cargo test -p app --lib --features dynamic-backends,vulkan -- --ignored --nocapture real_whisper`
    #[test]
    #[ignore]
    fn real_whisper_transcribes_a_wav() {
        let model = std::env::var("ASR_MODEL").expect("ASR_MODEL: path of a whisper ggml model");
        let wav = std::env::var("ASR_WAV").expect("ASR_WAV: path of a 16 kHz mono WAV");
        let mut reader = hound::WavReader::open(wav).expect("the WAV should open");
        assert_eq!(reader.spec().sample_rate, 16_000, "the WAV must be 16 kHz");
        let samples: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();

        let started = std::time::Instant::now();
        let mut whisper = Whisper::load(Path::new(&model)).expect("the model should load");
        eprintln!("loaded in {:.2}s", started.elapsed().as_secs_f64());
        let started = std::time::Instant::now();
        let text = whisper.transcribe(&samples, None).expect("transcription");
        eprintln!("TRANSCRIPT ({:.2}s for {:.1}s of audio): {text}", started.elapsed().as_secs_f64(), samples.len() as f64 / 16_000.0);
        assert!(text.to_lowercase().contains("country"), "unexpected: {text}");
    }
}
