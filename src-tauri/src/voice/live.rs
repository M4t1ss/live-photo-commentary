//! Voice input in the running app: what the controller's decisions do to the
//! real microphone, pipeline and speech recognizer, and the thread that runs
//! the controller.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use crate::config::Config;
use crate::events::{Event, EventSink};
use crate::state::AppPipeline;

use super::asr::AsrWorker;
use super::audio::{self, Mic};
use super::controller::{Controller, Effects, Input, Settings};

/// Sends [`Input`]s to the controller's thread. Cheap to clone.
#[derive(Clone)]
pub struct VoiceHandle {
    tx: mpsc::Sender<Input>,
}

impl VoiceHandle {
    pub fn send(&self, input: Input) {
        let _ = self.tx.send(input);
    }
}

/// Starts the controller on a thread of its own. It lives as long as the
/// app, and so does the microphone stream it opens, which stays on that
/// thread.
pub fn start(pipeline: Arc<AppPipeline>, asr: Arc<AsrWorker>, config: Arc<Mutex<Config>>, sink: EventSink) -> VoiceHandle {
    let (tx, rx) = mpsc::channel();
    let settings = Settings::from_config(&config.lock().expect("config mutex should not be poisoned"));
    let handle = VoiceHandle { tx: tx.clone() };
    let spawned = std::thread::Builder::new().name("voice".to_string()).spawn(move || {
        let effects =
            LiveEffects { pipeline, asr, config, sink, tx, mic: None, meter_frames: 0, newest_turn: Arc::default() };
        let mut controller = Controller::new(effects, settings);
        while let Ok(input) = rx.recv() {
            controller.handle(input);
        }
    });
    if let Err(e) = spawned {
        log::error!("cannot start the voice input thread: {e}");
    }
    handle
}

/// How many measurements go to the meter for each one that is sent on: the
/// microphone is measured 50 times a second, which is more than a meter needs.
const METER_EVERY: usize = 3;

struct LiveEffects {
    pipeline: Arc<AppPipeline>,
    asr: Arc<AsrWorker>,
    config: Arc<Mutex<Config>>,
    sink: EventSink,
    /// The controller's own input channel, for what comes back from the
    /// microphone and the recognizer.
    tx: mpsc::Sender<Input>,
    mic: Option<Mic>,
    meter_frames: usize,
    /// The number of the newest voice turn, so that a transcription that has
    /// been superseded is not started.
    newest_turn: Arc<AtomicU64>,
}

impl LiveEffects {
    fn config(&self) -> Config {
        self.config.lock().expect("config mutex should not be poisoned").clone()
    }
}

impl Effects for LiveEffects {
    fn open_mic(&mut self) -> Result<(), String> {
        let tx = self.tx.clone();
        let mic = Mic::open(self.config().voice_input_device.as_deref(), move |db| {
            let _ = tx.send(Input::Level(db));
        })?;
        if let Some(warning) = &mic.warning {
            self.error(warning.clone());
        }
        log::info!("microphone open at {} Hz", mic.sample_rate);
        self.mic = Some(mic);
        Ok(())
    }

    fn close_mic(&mut self) {
        if self.mic.take().is_some() {
            log::info!("microphone closed");
        }
    }

    fn start_recording(&mut self) {
        if let Some(mic) = &self.mic {
            mic.recorder.lock().expect("recorder mutex should not be poisoned").start();
        }
    }

    fn finish_recording(&mut self) -> Result<Vec<f32>, String> {
        let mic = self.mic.as_ref().ok_or("The microphone is not open")?;
        let samples = mic.recorder.lock().expect("recorder mutex should not be poisoned").finish();
        log::info!("recorded {:.1}s", samples.len() as f64 / f64::from(mic.sample_rate));
        audio::resample_to_16k(&samples, mic.sample_rate)
    }

    fn discard_recording(&mut self) {
        if let Some(mic) = &self.mic {
            mic.recorder.lock().expect("recorder mutex should not be poisoned").finish();
        }
    }

    fn begin_turn(&mut self) -> u64 {
        let turn = self.pipeline.voice_begin();
        self.newest_turn.store(turn, Ordering::SeqCst);
        log::info!("voice turn {turn} begins");
        turn
    }

    fn request_capture(&mut self, turn: u64) {
        self.pipeline.voice_request_capture(turn);
    }

    fn transcribing(&mut self, turn: u64) {
        self.pipeline.voice_transcribing(turn);
    }

    fn transcribe(&mut self, turn: u64, samples: Vec<f32>) {
        let language = Some(self.config().voice_language).filter(|l| l != "auto");
        let tx = self.tx.clone();
        let newest = Arc::clone(&self.newest_turn);
        self.asr.transcribe(
            samples,
            language,
            move || newest.load(Ordering::SeqCst) == turn,
            move |text| {
                let _ = tx.send(Input::Transcribed { turn, text });
            },
        );
    }

    fn set_transcript(&mut self, turn: u64, text: String) {
        log::info!("voice turn {turn}: {text:?}");
        self.pipeline.voice_set_transcript(&self.config(), turn, text);
    }

    fn abort_turn(&mut self, turn: u64) {
        log::info!("voice turn {turn} ends without an answer");
        self.pipeline.voice_abort(turn);
    }

    fn level(&mut self, db: f32) {
        self.meter_frames += 1;
        if self.meter_frames % METER_EVERY == 0 {
            (self.sink)(Event::VoiceLevel { db });
        }
    }

    fn error(&mut self, message: String) {
        log::warn!("voice input: {message}");
        (self.sink)(Event::Error { message });
    }
}
