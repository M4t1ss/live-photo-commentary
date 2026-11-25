//! The VLM/TTS pipeline: frame differencing, and the two
//! dedicated worker threads that own the describer and synthesizer. Jobs reach
//! a worker over an `mpsc` channel, and neither worker runs on Tauri's async
//! runtime or the UI thread, because llama.cpp and Kokoro block; each has its
//! own single-threaded Tokio runtime for the async APIs.
//!
//! There's no frame queue or auto-loop task: Tauri commands capture a
//! screenshot and hand it straight to [`Pipeline::trigger`], so
//! `start_cycle`/`stop_cycle` only need to reset state, not manage an async
//! consumer task. There's no on-disk audio store either: a WAV-encoded
//! chunk's bytes travel in its [`ChunkEvent`] and reach the frontend as a data
//! URL.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use image::DynamicImage;

use crate::config::Config;
use crate::difference;
use crate::events::{ChunkEvent, ChunkOrigin, Event, EventSink};
use crate::voice_turn::VoiceTurn;

// ── Describer/synthesizer traits ───────────────────────────────────────────

/// What the pipeline needs from a VLM describer. Implemented for
/// [`vlm_describer::Describer`]; tests implement it for fakes, so the
/// pipeline can be exercised without a model or network access.
pub trait DescriberOps: Send {
    fn set_max_history_size(&mut self, n: usize);
    fn set_prompts(&mut self, prompts: vlm_describer::Prompts);
    /// `options.cancel` returns true once the result would be discarded
    /// anyway (the generation moved on); the describer should then give up
    /// early. Its error is not reported, because the worker drops the results
    /// and errors of a stale generation. `options.user_text` is what the user
    /// said, for a voice turn.
    fn describe(
        &mut self,
        current: &DynamicImage,
        previous: Option<&DynamicImage>,
        options: &vlm_describer::DescribeOptions<'_>,
    ) -> impl Future<Output = Result<String, String>> + Send;
    fn generate(&mut self, prompt: &str, system_prompt: Option<&str>) -> impl Future<Output = Result<String, String>> + Send;
    fn reset(&mut self);
}

impl DescriberOps for vlm_describer::Describer {
    fn set_max_history_size(&mut self, n: usize) {
        self.max_history_size = n;
    }

    fn set_prompts(&mut self, prompts: vlm_describer::Prompts) {
        self.prompts = prompts;
    }

    async fn describe(
        &mut self,
        current: &DynamicImage,
        previous: Option<&DynamicImage>,
        options: &vlm_describer::DescribeOptions<'_>,
    ) -> Result<String, String> {
        vlm_describer::Describer::describe_with(self, current, previous, options).await.map_err(|e| e.to_string())
    }

    async fn generate(&mut self, prompt: &str, system_prompt: Option<&str>) -> Result<String, String> {
        vlm_describer::Describer::generate(self, prompt, system_prompt).await.map_err(|e| e.to_string())
    }

    fn reset(&mut self) {
        vlm_describer::Describer::reset(self);
    }
}

/// What the pipeline needs from a TTS synthesizer. Implemented for
/// [`kokoro_timestamped::KokoroSynthesizer`]; tests implement it for fakes.
pub trait SynthesizerOps: Send {
    #[allow(clippy::type_complexity)]
    fn synthesize<'a>(
        &'a mut self,
        text: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<kokoro_timestamped::Chunk, String>> + 'a>, String>;
}

impl SynthesizerOps for kokoro_timestamped::KokoroSynthesizer {
    fn synthesize<'a>(
        &'a mut self,
        text: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<kokoro_timestamped::Chunk, String>> + 'a>, String> {
        let iter = kokoro_timestamped::KokoroSynthesizer::synthesize(self, text).map_err(|e| e.to_string())?;
        Ok(Box::new(iter.map(|r| r.map_err(|e| e.to_string()))))
    }
}

// ── Jobs ────────────────────────────────────────────────────────────────────

enum VlmJob<D> {
    Frame {
        gen: u64,
        current: Arc<DynamicImage>,
        previous: Option<Arc<DynamicImage>>,
        max_history_size: usize,
        /// What the user said, for a voice turn.
        user_text: Option<String>,
    },
    SetDescriber(Option<D>, mpsc::Sender<()>),
    SetPrompts(vlm_describer::Prompts),
    Generate { prompt: String, system_prompt: Option<String>, reply: mpsc::Sender<Result<String, String>> },
    Reset,
    /// Replies once dequeued; lets a caller wait for everything sent before
    /// it to have been fully processed (tests only, see `flush_vlm`).
    #[cfg_attr(not(test), allow(dead_code))]
    Sync(mpsc::Sender<()>),
}

enum TtsJob<S> {
    Cycle { gen: u64, text: String },
    System { text: String },
    /// Synthesizes `text` fully and returns every chunk, instead of
    /// streaming them to the event sink (pre-generation).
    Collect { text: String, reply: mpsc::Sender<Result<Vec<ChunkEvent>, String>> },
    SetSynthesizer(Option<S>, mpsc::Sender<()>),
    /// See [`VlmJob::Sync`].
    #[cfg_attr(not(test), allow(dead_code))]
    Sync(mpsc::Sender<()>),
}

// ── WAV encoding ────────────────────────────────────────────────────────────

/// 16-bit PCM mono WAV at `KokoroSynthesizer::SAMPLE_RATE`: each sample is
/// scaled by 32767 and clipped to the 16-bit range.
fn encode_wav(samples: &[f32]) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: kokoro_timestamped::KokoroSynthesizer::SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("WAV spec should be valid");
        for &sample in samples {
            let scaled = (sample as f64 * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(scaled).expect("write WAV sample");
        }
        writer.finalize().expect("finalize WAV");
    }
    cursor.into_inner()
}

fn to_chunk_event(origin: ChunkOrigin, index: usize, chunk: kokoro_timestamped::Chunk) -> ChunkEvent {
    ChunkEvent {
        origin,
        index,
        text: chunk.text,
        wav: encode_wav(&chunk.audio).into(),
        phonemes: chunk.phonemes,
        words: chunk.words.map(|words| words.into_iter().map(Into::into).collect()),
        tags: chunk.tags,
    }
}

// ── Worker threads ──────────────────────────────────────────────────────────

fn run_vlm_worker<D, S>(
    rx: mpsc::Receiver<VlmJob<D>>,
    generation: Arc<AtomicU64>,
    pending_frame: Arc<AtomicBool>,
    pending_cycle_tts: Arc<AtomicU64>,
    tts_tx: mpsc::Sender<TtsJob<S>>,
    sink: EventSink,
) where
    D: DescriberOps,
    S: SynthesizerOps,
{
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread tokio runtime should build");
    let mut describer: Option<D> = None;

    while let Ok(job) = rx.recv() {
        match job {
            VlmJob::Frame { gen, current, previous, max_history_size, user_text } => {
                // Dequeuing frees the "queue of one" slot immediately, even
                // though describing is still ahead of us.
                pending_frame.store(false, Ordering::SeqCst);
                if gen != generation.load(Ordering::SeqCst) {
                    continue;
                }
                let Some(d) = describer.as_mut() else {
                    sink(Event::Error { message: "No VLM configured".to_string() });
                    continue;
                };
                d.set_max_history_size(max_history_size);
                // A newer generation (stop, or a new frame) makes this result
                // worthless, so the describer is told to stop generating it.
                let cancelled = || gen != generation.load(Ordering::SeqCst);
                let options = vlm_describer::DescribeOptions { cancel: Some(&cancelled), user_text: user_text.as_deref() };
                let result = rt.block_on(d.describe(&current, previous.as_deref(), &options));
                // Checked before the result, so a cancelled description (an
                // error) and a stale one are both dropped silently.
                if cancelled() {
                    continue;
                }
                let text = match result {
                    Ok(text) => text,
                    Err(e) => {
                        sink(Event::Error { message: format!("VLM error: {e}") });
                        continue;
                    }
                };
                // Busy only if a job of this very generation is already
                // queued. A job left by an older generation is dropped by the
                // TTS worker when it gets to it, so it must not block this one.
                if pending_cycle_tts.swap(gen, Ordering::SeqCst) == gen {
                    sink(Event::Error { message: "TTS busy".to_string() });
                } else {
                    let _ = tts_tx.send(TtsJob::Cycle { gen, text });
                }
            }
            VlmJob::SetDescriber(new, ack) => {
                describer = new;
                let _ = ack.send(());
            }
            VlmJob::SetPrompts(prompts) => {
                if let Some(d) = describer.as_mut() {
                    d.set_prompts(prompts);
                }
            }
            VlmJob::Generate { prompt, system_prompt, reply } => {
                let result = match describer.as_mut() {
                    Some(d) => rt.block_on(d.generate(&prompt, system_prompt.as_deref())),
                    None => Err("No VLM configured".to_string()),
                };
                let _ = reply.send(result);
            }
            VlmJob::Reset => {
                if let Some(d) = describer.as_mut() {
                    d.reset();
                }
            }
            VlmJob::Sync(ack) => {
                let _ = ack.send(());
            }
        }
    }
}

fn run_tts_worker<S: SynthesizerOps>(
    rx: mpsc::Receiver<TtsJob<S>>,
    generation: Arc<AtomicU64>,
    pending_cycle_tts: Arc<AtomicU64>,
    sink: EventSink,
) {
    let mut synthesizer: Option<S> = None;

    while let Ok(job) = rx.recv() {
        match job {
            TtsJob::Cycle { gen, text } => {
                // Clear the "queued" mark, unless a newer generation has set
                // its own since.
                let _ = pending_cycle_tts.compare_exchange(gen, 0, Ordering::SeqCst, Ordering::SeqCst);
                if gen != generation.load(Ordering::SeqCst) {
                    continue;
                }
                let Some(s) = synthesizer.as_mut() else {
                    sink(Event::Error { message: "No TTS configured".to_string() });
                    continue;
                };
                match s.synthesize(&text) {
                    Err(e) => sink(Event::Error { message: format!("TTS error: {e}") }),
                    Ok(iter) => {
                        let mut completed = true;
                        for (index, chunk) in iter.enumerate() {
                            if gen != generation.load(Ordering::SeqCst) {
                                completed = false;
                                break;
                            }
                            match chunk {
                                Err(e) => {
                                    sink(Event::Error { message: format!("TTS error: {e}") });
                                    completed = false;
                                    break;
                                }
                                Ok(chunk) => sink(Event::Chunk(to_chunk_event(ChunkOrigin::Cycle { gen }, index, chunk))),
                            }
                        }
                        if completed {
                            sink(Event::TtsDone { gen });
                        }
                    }
                }
            }
            TtsJob::System { text } => {
                if let Some(s) = synthesizer.as_mut() {
                    match s.synthesize(&text) {
                        Err(e) => sink(Event::Error { message: format!("TTS error: {e}") }),
                        Ok(iter) => {
                            for (index, chunk) in iter.enumerate() {
                                match chunk {
                                    Err(e) => {
                                        sink(Event::Error { message: format!("TTS error: {e}") });
                                        break;
                                    }
                                    Ok(chunk) => sink(Event::Chunk(to_chunk_event(ChunkOrigin::System, index, chunk))),
                                }
                            }
                        }
                    }
                }
                sink(Event::SystemTtsDone);
            }
            TtsJob::Collect { text, reply } => {
                let result = (|| -> Result<Vec<ChunkEvent>, String> {
                    let s = synthesizer.as_mut().ok_or_else(|| "No TTS configured".to_string())?;
                    let iter = s.synthesize(&text)?;
                    iter.enumerate()
                        .map(|(index, chunk)| chunk.map(|c| to_chunk_event(ChunkOrigin::System, index, c)))
                        .collect()
                })();
                let _ = reply.send(result);
            }
            TtsJob::SetSynthesizer(new, ack) => {
                synthesizer = new;
                let _ = ack.send(());
            }
            TtsJob::Sync(ack) => {
                let _ = ack.send(());
            }
        }
    }
}

// ── Pipeline ─────────────────────────────────────────────────────────────

struct FrameState {
    prev_image: Option<Arc<DynamicImage>>,
    last_rejected: bool,
}

/// A pre-generated system message with the prompts it was generated from, so
/// `pregen_system_messages` can tell whether it is still current.
struct PregenMessage {
    prompt: String,
    system_prompt: String,
    chunks: Vec<ChunkEvent>,
}

/// Frame differencing plus the VLM/TTS worker threads. Generic over the
/// describer/synthesizer traits so it can be driven by fakes in tests.
pub struct Pipeline<D, S> {
    generation: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
    frame_state: Mutex<FrameState>,
    /// The voice turn in progress, if any.
    voice: Mutex<VoiceTurn>,
    pending_frame: Arc<AtomicBool>,
    /// The generation of the cycle TTS job waiting in the TTS queue, or 0 for
    /// none. Only the clones handed to the workers are used in production;
    /// this copy exists so tests can simulate "already queued" directly (see
    /// the `tts_busy_error...` test).
    #[cfg_attr(not(test), allow(dead_code))]
    pending_cycle_tts: Arc<AtomicU64>,
    pregen_messages: Mutex<HashMap<String, PregenMessage>>,
    /// Bumped by every `pregen_system_messages` and `discard_pregen_messages`;
    /// a pre-generation that finds it changed has been superseded.
    pregen_epoch: AtomicU64,
    vlm_tx: mpsc::Sender<VlmJob<D>>,
    tts_tx: mpsc::Sender<TtsJob<S>>,
    sink: EventSink,
}

impl<D, S> Pipeline<D, S>
where
    D: DescriberOps + 'static,
    S: SynthesizerOps + 'static,
{
    /// Spawns the VLM and TTS worker threads, both starting without a model.
    pub fn new(sink: EventSink) -> Self {
        let generation = Arc::new(AtomicU64::new(0));
        let pending_frame = Arc::new(AtomicBool::new(false));
        let pending_cycle_tts = Arc::new(AtomicU64::new(0));

        let (vlm_tx, vlm_rx) = mpsc::channel::<VlmJob<D>>();
        let (tts_tx, tts_rx) = mpsc::channel::<TtsJob<S>>();

        {
            let generation = Arc::clone(&generation);
            let pending_frame = Arc::clone(&pending_frame);
            let pending_cycle_tts = Arc::clone(&pending_cycle_tts);
            let tts_tx = tts_tx.clone();
            let sink = Arc::clone(&sink);
            std::thread::spawn(move || run_vlm_worker(vlm_rx, generation, pending_frame, pending_cycle_tts, tts_tx, sink));
        }
        {
            let generation = Arc::clone(&generation);
            let pending_cycle_tts = Arc::clone(&pending_cycle_tts);
            let sink = Arc::clone(&sink);
            std::thread::spawn(move || run_tts_worker(tts_rx, generation, pending_cycle_tts, sink));
        }

        Self {
            generation,
            running: Arc::new(AtomicBool::new(false)),
            frame_state: Mutex::new(FrameState { prev_image: None, last_rejected: false }),
            voice: Mutex::new(VoiceTurn::default()),
            pending_frame,
            pending_cycle_tts,
            pregen_messages: Mutex::new(HashMap::new()),
            pregen_epoch: AtomicU64::new(0),
            vlm_tx,
            tts_tx,
            sink,
        }
    }

    pub fn running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Replaces the live describer. The VLM worker only picks this job up
    /// once it's done with whatever it was processing, so this blocks until
    /// any in-flight generation has actually finished, without needing a
    /// lock, since the worker thread already serializes every describer
    /// access.
    pub fn set_describer(&self, describer: Option<D>) {
        self.discard_pregen_messages();
        let (ack, rx) = mpsc::channel();
        if self.vlm_tx.send(VlmJob::SetDescriber(describer, ack)).is_ok() {
            let _ = rx.recv();
        }
    }

    /// Replaces the live synthesizer, waiting the same way as `set_describer`.
    pub fn set_synthesizer(&self, synthesizer: Option<S>) {
        self.discard_pregen_messages();
        let (ack, rx) = mpsc::channel();
        if self.tts_tx.send(TtsJob::SetSynthesizer(synthesizer, ack)).is_ok() {
            let _ = rx.recv();
        }
    }

    /// Pushes new prompts onto the live describer without a model reload; a
    /// no-op if there's no describer.
    pub fn set_prompts(&self, prompts: vlm_describer::Prompts) {
        let _ = self.vlm_tx.send(VlmJob::SetPrompts(prompts));
    }

    /// Forgets the describer's comment history without touching anything
    /// else.
    pub fn reset_describer_history(&self) {
        let _ = self.vlm_tx.send(VlmJob::Reset);
    }

    /// Diffs `image` against the previous frame, decides whether to push it
    /// to the VLM, and sends the `frame`/`skipped`/`busy` events. Ignored
    /// while a voice turn is in progress: a screenshot taken for the normal
    /// cycle that arrives late must not get ahead of the voice turn's.
    pub fn trigger(&self, cfg: &Config, image: DynamicImage) {
        if self.voice.lock().expect("voice mutex should not be poisoned").is_active() {
            return;
        }
        self.submit(cfg, image, None);
    }

    /// The common part of `trigger` and a finished voice turn. A voice turn
    /// (`user_text` is set) is always described: it skips the difference
    /// check, and is not dropped as "busy".
    fn submit(&self, cfg: &Config, image: DynamicImage, user_text: Option<String>) {
        let image = Arc::new(image);
        let mut state = self.frame_state.lock().expect("frame state mutex should not be poisoned");
        let previous = state.prev_image.clone();

        let mut diff = None;
        if let Some(prev) = &previous {
            if cfg.difference_threshold > 0.0 && user_text.is_none() {
                match difference::difference(&image, prev, &cfg.difference_measure) {
                    Ok(d) => diff = Some(d),
                    Err(message) => (self.sink)(Event::Error { message }),
                }
            }
        }
        let accepted = diff.is_none_or(|d| d >= cfg.difference_threshold);

        // Only push the displayed frame into "previous" when the currently
        // shown frame was itself accepted; consecutive rejections (and the
        // accepted frame that ends a rejection streak) replace it in place.
        let push = !state.last_rejected;
        state.last_rejected = !accepted;
        state.prev_image = Some(Arc::clone(&image));
        drop(state);

        let gen = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        (self.sink)(Event::Frame {
            image: Arc::clone(&image),
            push,
            gen,
            diff,
            measure: diff.map(|_| cfg.difference_measure.clone()),
        });

        if !accepted {
            (self.sink)(Event::Skipped { diff: diff.expect("a rejected frame should have a diff") });
            (self.sink)(Event::TakeScreenshot);
            return;
        }

        // A voice turn has bumped the generation (`interrupt`), so a job still
        // queued from before is dropped by the worker; it is not "busy".
        if self.pending_frame.swap(true, Ordering::SeqCst) && user_text.is_none() {
            (self.sink)(Event::Busy { message: "VLM busy, cycle skipped".to_string() });
            return;
        }
        let _ = self.vlm_tx.send(VlmJob::Frame {
            gen,
            current: image,
            previous,
            max_history_size: cfg.max_history_size as usize,
            user_text,
        });
    }

    /// Stops what the avatar is doing for the user to speak: the generation
    /// moves on, so a description in progress is abandoned (see
    /// `DescriberOps::describe`) and no more speech of the old turn is sent.
    /// Speech already being synthesized finishes, and is dropped.
    pub fn interrupt(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.pending_frame.store(false, Ordering::SeqCst);
    }

    // ── Voice turns ──
    // The voice controller drives these; a turn ends in `submit` (both
    // parts arrived), `voice_abort`, or a stop of the cycle.

    /// Starts a voice turn, interrupting whatever is going on, and returns its
    /// number. Sends `voice_listening`.
    #[allow(dead_code)] // called by the voice controller
    pub fn voice_begin(&self) -> u64 {
        self.interrupt();
        let turn = self.voice.lock().expect("voice mutex should not be poisoned").begin();
        (self.sink)(Event::VoiceListening { turn });
        turn
    }

    /// Asks the frontend to take this turn's screenshot now (`voice_capture`).
    /// The controller calls this when the screenshot is wanted, which decides
    /// whether it shows the screen when the user starts or when they finish.
    #[allow(dead_code)] // called by the voice controller
    pub fn voice_request_capture(&self, turn: u64) {
        if self.voice.lock().expect("voice mutex should not be poisoned").is_current(turn) {
            (self.sink)(Event::VoiceCapture { turn });
        }
    }

    /// Tells the frontend the recording ended and is being transcribed.
    #[allow(dead_code)] // called by the voice controller
    pub fn voice_transcribing(&self, turn: u64) {
        if self.voice.lock().expect("voice mutex should not be poisoned").is_current(turn) {
            (self.sink)(Event::VoiceTranscribing);
        }
    }

    /// The screenshot taken for `turn`. If the transcript is already there,
    /// the turn is described.
    pub fn voice_set_frame(&self, cfg: &Config, turn: u64, image: DynamicImage) {
        let ready = self.voice.lock().expect("voice mutex should not be poisoned").set_frame(turn, image);
        if let Some(ready) = ready {
            self.submit(cfg, ready.image, Some(ready.text));
        }
    }

    /// What the user said in `turn`. Sends `voice_transcript`; if the
    /// screenshot is already there, the turn is described. Nothing said
    /// (only whitespace) ends the turn like `voice_abort`.
    #[allow(dead_code)] // called by the voice controller
    pub fn voice_set_transcript(&self, cfg: &Config, turn: u64, text: String) {
        let text = text.trim().to_string();
        if text.is_empty() {
            self.voice_abort(turn);
            return;
        }
        let ready = {
            let mut voice = self.voice.lock().expect("voice mutex should not be poisoned");
            if !voice.is_current(turn) {
                return;
            }
            voice.set_transcript(turn, text.clone())
        };
        (self.sink)(Event::VoiceTranscript { text });
        if let Some(ready) = ready {
            self.submit(cfg, ready.image, Some(ready.text));
        }
    }

    /// Ends `turn` without an answer (nothing recognised, or it failed): the
    /// screenshot it may have taken is dropped, and `voice_empty` tells the
    /// frontend to resume its normal screenshot timing.
    #[allow(dead_code)] // called by the voice controller
    pub fn voice_abort(&self, turn: u64) {
        if self.voice.lock().expect("voice mutex should not be poisoned").abort(turn) {
            (self.sink)(Event::VoiceEmpty);
        }
    }

    /// Resets the frame-diffing state and the describer's history for a
    /// fresh cycle; frames arrive directly through `trigger`.
    pub fn start_cycle(&self) {
        self.running.store(true, Ordering::SeqCst);
        let mut state = self.frame_state.lock().expect("frame state mutex should not be poisoned");
        state.prev_image = None;
        state.last_rejected = false;
        drop(state);
        let _ = self.vlm_tx.send(VlmJob::Reset);
    }

    /// Stops the cycle and bumps the generation, so in-flight VLM/TTS work
    /// is dropped once the workers reach it.
    pub fn stop_cycle(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.voice.lock().expect("voice mutex should not be poisoned").reset();
    }

    /// Synthesizes arbitrary text and streams it as system chunks, always
    /// ending with `system_tts_done`.
    pub fn synthesize_system(&self, text: String) {
        let _ = self.tts_tx.send(TtsJob::System { text });
    }

    /// Replays previously pre-generated chunks for `name`, always ending
    /// with `system_tts_done`.
    pub fn play_system_message(&self, name: &str) {
        let chunks = self.pregen_messages.lock().expect("pregen mutex should not be poisoned").get(name).map(|m| m.chunks.clone());
        for chunk in chunks.into_iter().flatten() {
            (self.sink)(Event::Chunk(chunk));
        }
        (self.sink)(Event::SystemTtsDone);
    }

    /// Generates and synthesizes each non-empty prompt in `prompts` in turn
    /// (keys are message names, e.g. `"greeting"`), storing the finished
    /// chunks for later replay by `play_system_message`, then calls
    /// `on_done`. A stored message made from the same prompt and
    /// `system_prompt` is kept as it is, until `discard_pregen_messages`. A
    /// newer call, or a discard, cancels this one: it stops at the next
    /// message and stores nothing more (`on_done` is still called). Runs on
    /// its own thread, driving the VLM/TTS workers through their job queues
    /// like everything else.
    pub fn pregen_system_messages(
        self: Arc<Self>,
        prompts: HashMap<String, String>,
        system_prompt: String,
        on_done: impl FnOnce() + Send + 'static,
    ) {
        let epoch = self.pregen_epoch.fetch_add(1, Ordering::SeqCst) + 1;
        std::thread::spawn(move || {
            let is_current = |name: &str, prompt: &str, stored: &HashMap<String, PregenMessage>| {
                stored.get(name).is_some_and(|m| m.prompt == prompt && m.system_prompt == system_prompt)
            };
            // Drop everything that changed (or is no longer asked for) up front,
            // so stale clips don't survive if one generation fails mid-way.
            self.prune_pregen_messages(&prompts, &system_prompt);
            for (name, prompt) in prompts {
                if self.pregen_epoch.load(Ordering::SeqCst) != epoch {
                    break;
                }
                if prompt.trim().is_empty() {
                    continue;
                }
                if is_current(&name, &prompt, &self.pregen_messages.lock().expect("pregen mutex should not be poisoned")) {
                    continue;
                }
                let (reply, rx) = mpsc::channel();
                let job = VlmJob::Generate { prompt: prompt.clone(), system_prompt: Some(system_prompt.clone()), reply };
                if self.vlm_tx.send(job).is_err() {
                    continue;
                }
                let text = match rx.recv() {
                    Ok(Ok(text)) if !text.is_empty() => text,
                    Ok(Err(message)) => {
                        log::warn!("System message generation for {name:?} failed: {message}");
                        continue;
                    }
                    _ => continue,
                };

                let (reply, rx) = mpsc::channel();
                if self.tts_tx.send(TtsJob::Collect { text, reply }).is_err() {
                    continue;
                }
                match rx.recv() {
                    Ok(Ok(chunks)) => {
                        let message = PregenMessage { prompt, system_prompt: system_prompt.clone(), chunks };
                        let mut stored = self.pregen_messages.lock().expect("pregen mutex should not be poisoned");
                        if self.pregen_epoch.load(Ordering::SeqCst) == epoch {
                            stored.insert(name, message);
                        }
                    }
                    Ok(Err(message)) => log::warn!("System message synthesis for {name:?} failed: {message}"),
                    Err(_) => {}
                }
            }
            on_done();
        });
    }

    /// Drops the stored messages that no longer match `prompts` and
    /// `system_prompt` (changed, blanked or no longer asked for), so a stale
    /// clip can't be played. Returns whether any non-blank prompt is left
    /// without its message, i.e. whether `pregen_system_messages` has
    /// anything to generate.
    pub fn prune_pregen_messages(&self, prompts: &HashMap<String, String>, system_prompt: &str) -> bool {
        let mut stored = self.pregen_messages.lock().expect("pregen mutex should not be poisoned");
        stored.retain(|name, m| prompts.get(name).is_some_and(|prompt| m.prompt == *prompt && m.system_prompt == system_prompt));
        prompts.iter().any(|(name, prompt)| !prompt.trim().is_empty() && !stored.contains_key(name))
    }

    /// Forgets every pre-generated message and cancels a pre-generation in
    /// progress, so the next `pregen_system_messages` generates them all.
    pub fn discard_pregen_messages(&self) {
        self.pregen_epoch.fetch_add(1, Ordering::SeqCst);
        self.pregen_messages.lock().expect("pregen mutex should not be poisoned").clear();
    }

    /// Blocks until every job sent to the VLM worker before this call has
    /// been fully processed. Test-only: production code reacts to events,
    /// it doesn't need to wait for the pipeline.
    #[cfg(test)]
    fn flush_vlm(&self) {
        let (ack, rx) = mpsc::channel();
        if self.vlm_tx.send(VlmJob::Sync(ack)).is_ok() {
            let _ = rx.recv();
        }
    }

    /// See `flush_vlm`.
    #[cfg(test)]
    fn flush_tts(&self) {
        let (ack, rx) = mpsc::channel();
        if self.tts_tx.send(TtsJob::Sync(ack)).is_ok() {
            let _ = rx.recv();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    struct FakeDescriber {
        reply: Result<String, String>,
        /// If set, `describe` blocks on it until the test sends, letting a
        /// test control exactly when an in-flight generation "finishes".
        gate: Option<mpsc::Receiver<()>>,
        calls: Arc<AtomicU64>,
        /// If set, `describe` signals it, then waits until the cancel
        /// function fires (see `until_cancelled`).
        started: Option<mpsc::Sender<()>>,
        /// Set by `describe` when it saw the cancel function fire.
        saw_cancel: Arc<AtomicBool>,
        /// The `user_text` of each `describe` call, in order.
        seen_user_text: Arc<Mutex<Vec<Option<String>>>>,
    }

    impl FakeDescriber {
        fn new(reply: Result<String, String>, gate: Option<mpsc::Receiver<()>>) -> Self {
            Self {
                reply,
                gate,
                calls: Arc::new(AtomicU64::new(0)),
                started: None,
                saw_cancel: Arc::new(AtomicBool::new(false)),
                seen_user_text: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn ok(text: &str) -> Self {
            Self::new(Ok(text.to_string()), None)
        }

        fn err(message: &str) -> Self {
            Self::new(Err(message.to_string()), None)
        }

        fn gated(text: &str) -> (Self, mpsc::Sender<()>) {
            let (tx, rx) = mpsc::channel();
            (Self::new(Ok(text.to_string()), Some(rx)), tx)
        }

        /// A describer that, like a model generating tokens, runs until its
        /// cancel function fires; the receiver tells when it has started.
        fn until_cancelled() -> (Self, mpsc::Receiver<()>) {
            let (tx, rx) = mpsc::channel();
            let mut describer = Self::ok("never delivered");
            describer.started = Some(tx);
            (describer, rx)
        }
    }

    impl DescriberOps for FakeDescriber {
        fn set_max_history_size(&mut self, _n: usize) {}
        fn set_prompts(&mut self, _prompts: vlm_describer::Prompts) {}

        async fn describe(
            &mut self,
            _current: &DynamicImage,
            _previous: Option<&DynamicImage>,
            options: &vlm_describer::DescribeOptions<'_>,
        ) -> Result<String, String> {
            let never = || false;
            let cancelled = options.cancel.unwrap_or(&never);
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.seen_user_text.lock().unwrap().push(options.user_text.map(str::to_string));
            if let Some(started) = &self.started {
                let _ = started.send(());
                let give_up = Instant::now() + Duration::from_secs(5);
                while !cancelled() {
                    if Instant::now() > give_up {
                        return Ok("the cancel function never fired".to_string());
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                self.saw_cancel.store(true, Ordering::SeqCst);
                return Err("cancelled".to_string());
            }
            if let Some(gate) = &self.gate {
                let _ = gate.recv();
            }
            self.reply.clone()
        }

        async fn generate(&mut self, _prompt: &str, _system_prompt: Option<&str>) -> Result<String, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.reply.clone()
        }

        fn reset(&mut self) {}
    }

    struct FakeSynthesizer {
        chunks: Vec<Result<kokoro_timestamped::Chunk, String>>,
        /// If set, `next()` blocks on this right before yielding the
        /// *second* chunk, letting a test observe the first chunk's event
        /// before deciding what happens to the rest of the stream.
        gate_before_second: Option<mpsc::Receiver<()>>,
    }

    impl FakeSynthesizer {
        fn empty() -> Self {
            Self { chunks: Vec::new(), gate_before_second: None }
        }

        fn with_chunks(chunks: Vec<&str>) -> Self {
            Self {
                chunks: chunks.into_iter().map(|t| Ok(fake_chunk(t))).collect(),
                gate_before_second: None,
            }
        }

        fn gated_after_first(chunks: Vec<&str>) -> (Self, mpsc::Sender<()>) {
            let (tx, rx) = mpsc::channel();
            let synth =
                Self { chunks: chunks.into_iter().map(|t| Ok(fake_chunk(t))).collect(), gate_before_second: Some(rx) };
            (synth, tx)
        }
    }

    struct FakeChunkIter {
        chunks: std::vec::IntoIter<Result<kokoro_timestamped::Chunk, String>>,
        gate: Option<mpsc::Receiver<()>>,
        yielded: usize,
    }

    impl Iterator for FakeChunkIter {
        type Item = Result<kokoro_timestamped::Chunk, String>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.yielded == 1 {
                if let Some(gate) = &self.gate {
                    let _ = gate.recv();
                }
            }
            self.yielded += 1;
            self.chunks.next()
        }
    }

    impl SynthesizerOps for FakeSynthesizer {
        fn synthesize<'a>(
            &'a mut self,
            _text: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<kokoro_timestamped::Chunk, String>> + 'a>, String> {
            let chunks = std::mem::take(&mut self.chunks).into_iter();
            let gate = self.gate_before_second.take();
            Ok(Box::new(FakeChunkIter { chunks, gate, yielded: 0 }))
        }
    }

    fn fake_chunk(text: &str) -> kokoro_timestamped::Chunk {
        kokoro_timestamped::Chunk {
            audio: vec![0.0_f32; 4],
            text: text.to_string(),
            phonemes: vec![('a', 0.1)],
            words: None,
            tags: Vec::new(),
        }
    }

    fn test_image() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::new(4, 4))
    }

    fn capturing_sink() -> (EventSink, Arc<Mutex<Vec<Event>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let sink: EventSink = Arc::new(move |event: Event| captured.lock().unwrap().push(event));
        (sink, events)
    }

    fn wait_until(mut condition: impl FnMut() -> bool, timeout: Duration) {
        let start = Instant::now();
        while !condition() {
            assert!(start.elapsed() <= timeout, "condition not met within {timeout:?}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn new_pipeline(sink: EventSink) -> Pipeline<FakeDescriber, FakeSynthesizer> {
        Pipeline::new(sink)
    }

    #[test]
    fn first_frame_has_no_previous_and_is_pushed_and_sent_to_the_vlm() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        pipeline.set_describer(Some(FakeDescriber::ok("a comment")));
        pipeline.set_synthesizer(Some(FakeSynthesizer::empty()));

        pipeline.trigger(&Config::default(), test_image());
        pipeline.flush_vlm();

        let events = events.lock().unwrap();
        match &events[0] {
            Event::Frame { push, gen, diff, .. } => {
                assert!(push);
                assert_eq!(*gen, 1);
                assert_eq!(*diff, None);
            }
            other => panic!("expected a frame event, got {other:?}"),
        }
        assert!(!events.iter().any(|e| matches!(e, Event::Skipped { .. } | Event::Busy { .. })));
    }

    #[test]
    fn a_frame_below_the_threshold_is_rejected_and_requests_another_screenshot() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        pipeline.set_describer(Some(FakeDescriber::ok("a comment")));

        let cfg =
            Config { difference_threshold: 0.5, difference_measure: "mse".to_string(), ..Config::default() };

        // Two identical frames: mse is 0, below the 0.5 threshold.
        pipeline.trigger(&cfg, test_image());
        pipeline.trigger(&cfg, test_image());
        pipeline.flush_vlm();

        let events = events.lock().unwrap();
        let skipped: Vec<_> = events.iter().filter(|e| matches!(e, Event::Skipped { .. })).collect();
        assert_eq!(skipped.len(), 1, "only the second (diffed) frame should be skipped");
        assert!(events.iter().any(|e| matches!(e, Event::TakeScreenshot)));
    }

    #[test]
    fn push_stays_true_only_for_the_first_of_consecutive_rejections() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        pipeline.set_describer(Some(FakeDescriber::ok("a comment")));

        let cfg = Config { difference_threshold: 0.5, ..Config::default() };

        pipeline.trigger(&cfg, test_image()); // first frame: always pushed, always accepted (no previous)
        pipeline.trigger(&cfg, test_image()); // rejected (identical): push stays true (last wasn't rejected)
        pipeline.trigger(&cfg, test_image()); // rejected again: push is now false (last was rejected)
        pipeline.flush_vlm();

        let pushes: Vec<bool> = events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::Frame { push, .. } => Some(*push),
                _ => None,
            })
            .collect();
        assert_eq!(pushes, vec![true, true, false]);
    }

    #[test]
    fn vlm_busy_when_a_frame_is_already_pending() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        let describer = FakeDescriber::ok("a comment");
        let calls = Arc::clone(&describer.calls);
        pipeline.set_describer(Some(describer));

        // Simulate a frame already queued but not yet dequeued by the worker.
        pipeline.pending_frame.store(true, Ordering::SeqCst);
        pipeline.trigger(&Config::default(), test_image());
        pipeline.flush_vlm();

        assert_eq!(calls.load(Ordering::SeqCst), 0, "the busy frame must never reach the describer");
        let events = events.lock().unwrap();
        assert!(matches!(events.last(), Some(Event::Busy { .. })));
    }

    #[test]
    fn no_vlm_configured_reports_an_error() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);

        pipeline.trigger(&Config::default(), test_image());
        pipeline.flush_vlm();

        let events = events.lock().unwrap();
        assert!(events.iter().any(|e| matches!(e, Event::Error { message } if message == "No VLM configured")));
    }

    #[test]
    fn vlm_worker_drops_a_stale_generation_before_forwarding_to_tts() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        let (describer, release) = FakeDescriber::gated("a comment");
        pipeline.set_describer(Some(describer));
        pipeline.set_synthesizer(Some(FakeSynthesizer::empty()));

        // Starts the describe() call, which blocks until `release` fires.
        pipeline.trigger(&Config::default(), test_image());
        // The generation moves on while that call is still in flight.
        pipeline.stop_cycle();
        release.send(()).unwrap();
        pipeline.flush_vlm();
        pipeline.flush_tts();

        let events = events.lock().unwrap();
        assert!(
            !events.iter().any(|e| matches!(e, Event::Chunk(_) | Event::TtsDone { .. })),
            "a stale VLM result must never reach the TTS worker"
        );
        assert!(!events.iter().any(|e| matches!(e, Event::Error { message } if message.starts_with("VLM error"))));
    }

    #[test]
    fn vlm_error_is_reported_and_nothing_is_sent_to_tts() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        pipeline.set_describer(Some(FakeDescriber::err("boom")));
        pipeline.set_synthesizer(Some(FakeSynthesizer::empty()));

        pipeline.trigger(&Config::default(), test_image());
        pipeline.flush_vlm();

        let events = events.lock().unwrap();
        assert!(events.iter().any(|e| matches!(e, Event::Error { message } if message == "VLM error: boom")));
        assert!(!events.iter().any(|e| matches!(e, Event::Chunk(_))));
    }

    #[test]
    fn tts_busy_error_when_the_cycle_queue_is_already_full() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        pipeline.set_describer(Some(FakeDescriber::ok("a comment")));

        // Simulate a job of this very generation (the first, 1) already
        // queued ahead of this one.
        pipeline.pending_cycle_tts.store(1, Ordering::SeqCst);
        pipeline.trigger(&Config::default(), test_image());
        pipeline.flush_vlm();

        let events = events.lock().unwrap();
        assert!(events.iter().any(|e| matches!(e, Event::Error { message } if message == "TTS busy")));
    }

    #[test]
    fn a_cycle_tts_job_left_by_an_older_generation_does_not_make_the_next_one_busy() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        pipeline.set_describer(Some(FakeDescriber::ok("a comment")));
        pipeline.set_synthesizer(Some(FakeSynthesizer::with_chunks(vec!["one"])));

        // A job of an earlier generation is still queued, as after an
        // interruption while the TTS worker was busy.
        pipeline.pending_cycle_tts.store(99, Ordering::SeqCst);
        pipeline.trigger(&Config::default(), test_image());
        pipeline.flush_vlm();
        pipeline.flush_tts();

        let events = events.lock().unwrap();
        assert!(!events.iter().any(|e| matches!(e, Event::Error { message } if message == "TTS busy")));
        assert!(events.iter().any(|e| matches!(e, Event::TtsDone { .. })), "the new reply should be spoken");
    }

    #[test]
    fn what_the_user_said_reaches_the_describer_and_a_normal_frame_carries_none() {
        let (sink, _events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        let describer = FakeDescriber::ok("a comment");
        let seen = Arc::clone(&describer.seen_user_text);
        pipeline.set_describer(Some(describer));
        pipeline.set_synthesizer(Some(FakeSynthesizer::empty()));

        // A voice turn is built by the voice flow; here its job is sent directly.
        let gen = pipeline.generation.fetch_add(1, Ordering::SeqCst) + 1;
        pipeline
            .vlm_tx
            .send(VlmJob::Frame {
                gen,
                current: Arc::new(test_image()),
                previous: None,
                max_history_size: 0,
                user_text: Some("what is that?".to_string()),
            })
            .unwrap();
        pipeline.flush_vlm();
        pipeline.trigger(&Config::default(), test_image());
        pipeline.flush_vlm();

        assert_eq!(*seen.lock().unwrap(), [Some("what is that?".to_string()), None]);
    }

    // ── Voice turns ──

    /// A pipeline with a describer that records what the user said.
    fn voice_pipeline() -> (Pipeline<FakeDescriber, FakeSynthesizer>, Arc<Mutex<Vec<Event>>>, Arc<Mutex<Vec<Option<String>>>>) {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        let describer = FakeDescriber::ok("a comment");
        let seen = Arc::clone(&describer.seen_user_text);
        pipeline.set_describer(Some(describer));
        pipeline.set_synthesizer(Some(FakeSynthesizer::empty()));
        (pipeline, events, seen)
    }

    fn count(events: &Mutex<Vec<Event>>, matching: impl Fn(&Event) -> bool) -> usize {
        events.lock().unwrap().iter().filter(|e| matching(e)).count()
    }

    #[test]
    fn a_voice_turn_is_described_with_what_was_said_when_the_screenshot_comes_first() {
        let (pipeline, events, seen) = voice_pipeline();
        let cfg = Config::default();

        let turn = pipeline.voice_begin();
        pipeline.voice_set_frame(&cfg, turn, test_image());
        assert_eq!(count(&events, |e| matches!(e, Event::Frame { .. })), 0, "not before the user has spoken");
        pipeline.voice_set_transcript(&cfg, turn, "  what is that?  ".to_string());
        pipeline.flush_vlm();

        assert_eq!(*seen.lock().unwrap(), [Some("what is that?".to_string())]);
        let kinds: Vec<&str> = events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::VoiceListening { .. } => Some("listening"),
                Event::VoiceTranscript { text } if text == "what is that?" => Some("transcript"),
                Event::Frame { .. } => Some("frame"),
                _ => None,
            })
            .collect();
        assert_eq!(kinds, ["listening", "transcript", "frame"]);
    }

    #[test]
    fn a_voice_turn_is_described_when_the_transcript_comes_first() {
        let (pipeline, events, seen) = voice_pipeline();
        let cfg = Config::default();

        let turn = pipeline.voice_begin();
        pipeline.voice_set_transcript(&cfg, turn, "hello".to_string());
        assert_eq!(count(&events, |e| matches!(e, Event::Frame { .. })), 0, "not before the screenshot");
        pipeline.voice_set_frame(&cfg, turn, test_image());
        pipeline.flush_vlm();

        assert_eq!(*seen.lock().unwrap(), [Some("hello".to_string())]);
        assert_eq!(count(&events, |e| matches!(e, Event::Frame { .. })), 1);
    }

    #[test]
    fn a_voice_turn_is_described_even_if_the_screen_did_not_change_and_even_if_the_vlm_is_busy() {
        let (pipeline, events, seen) = voice_pipeline();
        let cfg = Config { difference_threshold: 0.5, ..Config::default() };
        pipeline.trigger(&cfg, test_image());
        pipeline.flush_vlm();

        // The same screen again: a normal frame would be skipped. And a frame is "pending".
        pipeline.pending_frame.store(true, Ordering::SeqCst);
        let turn = pipeline.voice_begin();
        pipeline.voice_set_frame(&cfg, turn, test_image());
        pipeline.voice_set_transcript(&cfg, turn, "hello".to_string());
        pipeline.flush_vlm();

        assert_eq!(*seen.lock().unwrap(), [None, Some("hello".to_string())]);
        assert_eq!(count(&events, |e| matches!(e, Event::Skipped { .. } | Event::Busy { .. })), 0);
    }

    #[test]
    fn a_regular_screenshot_arriving_during_a_voice_turn_is_ignored() {
        let (pipeline, events, seen) = voice_pipeline();
        let cfg = Config::default();

        let turn = pipeline.voice_begin();
        pipeline.trigger(&cfg, test_image()); // taken for the normal cycle before the key press
        pipeline.flush_vlm();
        assert_eq!(count(&events, |e| matches!(e, Event::Frame { .. })), 0);
        assert!(seen.lock().unwrap().is_empty());

        // Once the turn is over, regular screenshots count again.
        pipeline.voice_abort(turn);
        pipeline.trigger(&cfg, test_image());
        pipeline.flush_vlm();
        assert_eq!(*seen.lock().unwrap(), [None]);
    }

    #[test]
    fn beginning_a_voice_turn_cancels_a_description_in_flight_without_an_error() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        let (describer, started) = FakeDescriber::until_cancelled();
        let saw_cancel = Arc::clone(&describer.saw_cancel);
        pipeline.set_describer(Some(describer));
        pipeline.set_synthesizer(Some(FakeSynthesizer::with_chunks(vec!["one"])));

        pipeline.trigger(&Config::default(), test_image());
        started.recv_timeout(Duration::from_secs(2)).expect("the describer should start");
        pipeline.voice_begin();
        pipeline.flush_vlm();
        pipeline.flush_tts();

        assert!(saw_cancel.load(Ordering::SeqCst), "the describer should have been told to stop");
        assert_eq!(count(&events, |e| matches!(e, Event::Error { .. } | Event::Chunk(_) | Event::TtsDone { .. })), 0);
    }

    #[test]
    fn an_aborted_voice_turn_says_so_and_drops_the_screenshot() {
        let (pipeline, events, seen) = voice_pipeline();
        let cfg = Config::default();

        let turn = pipeline.voice_begin();
        pipeline.voice_set_frame(&cfg, turn, test_image()); // taken at the start
        pipeline.voice_abort(turn); // nothing was recognised
        pipeline.voice_set_transcript(&cfg, turn, "too late".to_string());
        pipeline.flush_vlm();

        assert_eq!(count(&events, |e| matches!(e, Event::VoiceEmpty)), 1);
        assert_eq!(count(&events, |e| matches!(e, Event::Frame { .. } | Event::VoiceTranscript { .. })), 0);
        assert!(seen.lock().unwrap().is_empty());
    }

    #[test]
    fn a_transcript_with_nothing_in_it_ends_the_turn() {
        let (pipeline, events, seen) = voice_pipeline();
        let turn = pipeline.voice_begin();
        pipeline.voice_set_transcript(&Config::default(), turn, " \n ".to_string());
        pipeline.flush_vlm();

        assert_eq!(count(&events, |e| matches!(e, Event::VoiceEmpty)), 1);
        assert_eq!(count(&events, |e| matches!(e, Event::VoiceTranscript { .. })), 0);
        assert!(seen.lock().unwrap().is_empty());
    }

    #[test]
    fn a_new_voice_turn_supersedes_the_one_before() {
        let (pipeline, events, seen) = voice_pipeline();
        let cfg = Config::default();

        let first = pipeline.voice_begin();
        let second = pipeline.voice_begin();
        // Only the turn in progress can be asked for a screenshot.
        pipeline.voice_request_capture(first);
        assert_eq!(count(&events, |e| matches!(e, Event::VoiceCapture { .. })), 0);
        pipeline.voice_request_capture(second);
        assert_eq!(count(&events, |e| matches!(e, Event::VoiceCapture { turn } if *turn == second)), 1);

        // Late parts of the first turn change nothing.
        pipeline.voice_set_frame(&cfg, first, test_image());
        pipeline.voice_set_transcript(&cfg, first, "old".to_string());
        pipeline.flush_vlm();
        assert!(seen.lock().unwrap().is_empty());
        assert_eq!(count(&events, |e| matches!(e, Event::VoiceTranscript { .. })), 0);
    }

    #[test]
    fn stopping_the_cycle_ends_the_voice_turn() {
        let (pipeline, _events, seen) = voice_pipeline();
        let cfg = Config::default();

        let turn = pipeline.voice_begin();
        pipeline.stop_cycle();
        pipeline.voice_set_frame(&cfg, turn, test_image());
        pipeline.voice_set_transcript(&cfg, turn, "hello".to_string());
        pipeline.flush_vlm();

        assert!(seen.lock().unwrap().is_empty());
    }

    #[test]
    fn a_new_generation_cancels_a_description_in_flight_without_an_error() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        let (describer, started) = FakeDescriber::until_cancelled();
        let saw_cancel = Arc::clone(&describer.saw_cancel);
        pipeline.set_describer(Some(describer));
        pipeline.set_synthesizer(Some(FakeSynthesizer::with_chunks(vec!["one"])));

        pipeline.trigger(&Config::default(), test_image());
        started.recv_timeout(Duration::from_secs(2)).expect("the describer should start");
        pipeline.stop_cycle();
        // Returns once the worker is free, which it is only after the
        // describer gave up.
        pipeline.flush_vlm();
        pipeline.flush_tts();

        assert!(saw_cancel.load(Ordering::SeqCst), "the describer should have been told to stop");
        let events = events.lock().unwrap();
        assert!(!events.iter().any(|e| matches!(e, Event::Error { .. })), "a cancellation is not an error: {events:?}");
        assert!(!events.iter().any(|e| matches!(e, Event::Chunk(_) | Event::TtsDone { .. })));
    }

    #[test]
    fn no_tts_configured_reports_an_error() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        pipeline.set_describer(Some(FakeDescriber::ok("a comment")));
        // No synthesizer set.

        pipeline.trigger(&Config::default(), test_image());
        pipeline.flush_vlm();
        pipeline.flush_tts();

        let events = events.lock().unwrap();
        assert!(events.iter().any(|e| matches!(e, Event::Error { message } if message == "No TTS configured")));
    }

    #[test]
    fn tts_worker_drops_mid_stream_when_the_generation_changes_and_skips_tts_done() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        pipeline.set_describer(Some(FakeDescriber::ok("a comment")));
        let (synthesizer, release_second) = FakeSynthesizer::gated_after_first(vec!["one", "two"]);
        pipeline.set_synthesizer(Some(synthesizer));

        pipeline.trigger(&Config::default(), test_image());
        pipeline.flush_vlm(); // the Cycle job is now queued for the TTS worker

        // Wait for the first chunk's event: by the time it's there, the TTS
        // worker is blocked fetching the second chunk (the gate), so
        // cancelling now is guaranteed to land between the two chunks.
        wait_until(|| events.lock().unwrap().iter().any(|e| matches!(e, Event::Chunk(_))), Duration::from_secs(2));
        pipeline.stop_cycle();
        release_second.send(()).unwrap();
        pipeline.flush_tts();

        let events = events.lock().unwrap();
        let chunk_texts: Vec<&str> =
            events.iter().filter_map(|e| match e { Event::Chunk(c) => Some(c.text.as_str()), _ => None }).collect();
        assert_eq!(chunk_texts, vec!["one"], "only the chunk sent before cancellation should arrive");
        assert!(!events.iter().any(|e| matches!(e, Event::TtsDone { .. })), "tts_done must not follow a cancelled run");
    }

    #[test]
    fn a_completed_cycle_ends_with_tts_done() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        pipeline.set_describer(Some(FakeDescriber::ok("a comment")));
        pipeline.set_synthesizer(Some(FakeSynthesizer::with_chunks(vec!["one", "two"])));

        pipeline.trigger(&Config::default(), test_image());
        pipeline.flush_vlm();
        pipeline.flush_tts();

        let events = events.lock().unwrap();
        let chunk_texts: Vec<&str> =
            events.iter().filter_map(|e| match e { Event::Chunk(c) => Some(c.text.as_str()), _ => None }).collect();
        assert_eq!(chunk_texts, vec!["one", "two"]);
        assert!(matches!(events.last(), Some(Event::TtsDone { gen: 1 })));
    }

    #[test]
    fn synthesize_system_streams_chunks_then_ends_with_system_tts_done() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);
        pipeline.set_synthesizer(Some(FakeSynthesizer::with_chunks(vec!["hello"])));

        pipeline.synthesize_system("hello".to_string());
        pipeline.flush_tts();

        let events = events.lock().unwrap();
        assert!(matches!(&events[0], Event::Chunk(c) if c.text == "hello" && c.origin == ChunkOrigin::System));
        assert!(matches!(events[1], Event::SystemTtsDone));
    }

    #[test]
    fn synthesize_system_without_a_synthesizer_still_ends_with_system_tts_done() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);

        pipeline.synthesize_system("hello".to_string());
        pipeline.flush_tts();

        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Event::SystemTtsDone));
    }

    #[test]
    fn play_system_message_replays_stored_chunks_then_system_tts_done() {
        let (sink, events) = capturing_sink();
        let pipeline = Arc::new(new_pipeline(sink));
        pipeline.set_describer(Some(FakeDescriber::ok("hi there")));
        pipeline.set_synthesizer(Some(FakeSynthesizer::with_chunks(vec!["hi there"])));

        let (done_tx, done_rx) = mpsc::channel();
        let prompts = HashMap::from([("greeting".to_string(), "say hi".to_string())]);
        Arc::clone(&pipeline).pregen_system_messages(prompts, "be nice".to_string(), move || {
            let _ = done_tx.send(());
        });
        done_rx.recv_timeout(Duration::from_secs(2)).expect("pregen should finish");
        events.lock().unwrap().clear(); // only interested in play_system_message's own events

        pipeline.play_system_message("greeting");

        let events = events.lock().unwrap();
        assert!(matches!(&events[0], Event::Chunk(c) if c.text == "hi there"));
        assert!(matches!(events[1], Event::SystemTtsDone));
    }

    #[test]
    fn play_system_message_with_no_stored_message_just_sends_system_tts_done() {
        let (sink, events) = capturing_sink();
        let pipeline = new_pipeline(sink);

        pipeline.play_system_message("never-pregenerated");

        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Event::SystemTtsDone));
    }

    #[test]
    fn pregen_skips_blank_prompts_and_generation_failures() {
        let (sink, _events) = capturing_sink();
        let pipeline = Arc::new(new_pipeline(sink));
        pipeline.set_describer(Some(FakeDescriber::err("no model")));
        pipeline.set_synthesizer(Some(FakeSynthesizer::empty()));

        let (done_tx, done_rx) = mpsc::channel();
        let prompts = HashMap::from([
            ("greeting".to_string(), "  ".to_string()), // blank: skipped before even asking the VLM
            ("farewell".to_string(), "say bye".to_string()), // VLM fails: skipped
        ]);
        Arc::clone(&pipeline).pregen_system_messages(prompts, "be nice".to_string(), move || {
            let _ = done_tx.send(());
        });
        done_rx.recv_timeout(Duration::from_secs(2)).expect("pregen should finish");

        assert!(pipeline.pregen_messages.lock().unwrap().is_empty());
    }

    #[test]
    fn pregen_only_regenerates_messages_whose_prompts_changed() {
        let (sink, _events) = capturing_sink();
        let pipeline = Arc::new(new_pipeline(sink));
        let describer = FakeDescriber::ok("hi there");
        let calls = Arc::clone(&describer.calls);
        pipeline.set_describer(Some(describer));
        pipeline.set_synthesizer(Some(FakeSynthesizer::with_chunks(vec!["hi there"])));

        let pregen = |greeting: &str, farewell: &str, system_prompt: &str| {
            let (done_tx, done_rx) = mpsc::channel();
            let prompts = HashMap::from([
                ("greeting".to_string(), greeting.to_string()),
                ("farewell".to_string(), farewell.to_string()),
            ]);
            Arc::clone(&pipeline).pregen_system_messages(prompts, system_prompt.to_string(), move || {
                let _ = done_tx.send(());
            });
            done_rx.recv_timeout(Duration::from_secs(2)).expect("pregen should finish");
            calls.load(Ordering::SeqCst)
        };

        assert_eq!(pregen("say hi", "say bye", "be nice"), 2, "first run generates both");
        assert_eq!(pregen("say hi", "say bye", "be nice"), 2, "nothing changed: nothing generated");
        assert_eq!(pregen("say hello", "say bye", "be nice"), 3, "only the greeting changed");
        assert_eq!(pregen("say hello", "  ", "be nice"), 3, "a blanked prompt is dropped, not generated");
        assert!(!pipeline.pregen_messages.lock().unwrap().contains_key("farewell"));
        assert_eq!(pregen("say hello", "say bye", "be kind"), 5, "a new system prompt changes every message");

        pipeline.discard_pregen_messages(); // a new avatar model, even with identical prompts
        assert_eq!(pregen("say hello", "say bye", "be kind"), 7, "after a discard everything is generated again");

        let prompts = |greeting: &str, farewell: &str| {
            HashMap::from([("greeting".to_string(), greeting.to_string()), ("farewell".to_string(), farewell.to_string())])
        };
        let stored = |name: &str| pipeline.pregen_messages.lock().unwrap().contains_key(name);
        assert!(!pipeline.prune_pregen_messages(&prompts("say hello", "say bye"), "be kind"), "nothing to generate");
        assert!(stored("greeting") && stored("farewell"));
        assert!(pipeline.prune_pregen_messages(&prompts("say hi", "say bye"), "be kind"), "a prompt changed");
        assert!(!stored("greeting") && stored("farewell"), "only the stale message is dropped");
        assert!(!pipeline.prune_pregen_messages(&prompts(" ", "say bye"), "be kind"), "a blanked prompt needs no generation");
        assert!(pipeline.prune_pregen_messages(&prompts("say hi", "say bye"), "be nice"), "the system prompt changed");
        assert!(!stored("farewell"));
    }

    #[test]
    fn swapping_the_describer_discards_pregenerated_messages() {
        let (sink, _events) = capturing_sink();
        let pipeline = Arc::new(new_pipeline(sink));
        pipeline.set_describer(Some(FakeDescriber::ok("hi there")));
        pipeline.set_synthesizer(Some(FakeSynthesizer::with_chunks(vec!["hi there"])));

        let (done_tx, done_rx) = mpsc::channel();
        let prompts = HashMap::from([("greeting".to_string(), "say hi".to_string())]);
        Arc::clone(&pipeline).pregen_system_messages(prompts, "be nice".to_string(), move || {
            let _ = done_tx.send(());
        });
        done_rx.recv_timeout(Duration::from_secs(2)).expect("pregen should finish");
        assert_eq!(pipeline.pregen_messages.lock().unwrap().len(), 1);

        pipeline.set_describer(Some(FakeDescriber::ok("other")));
        assert!(pipeline.pregen_messages.lock().unwrap().is_empty());
    }

    /// Runs the real pipeline (real `vlm_describer::Describer` and
    /// `kokoro_timestamped::KokoroSynthesizer`, no fakes) on the two sample
    /// frames, printing every event. Needs the local-only
    /// `tests/data/frame_{0,1}.png` and a `GEMINI_API_KEY`, so it only
    /// runs on request: `cargo test -- --ignored real_pipeline`.
    #[tokio::test]
    #[ignore]
    async fn real_pipeline_describes_and_synthesizes_the_sample_frames() {
        let frames_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let (frame0, frame1) = (frames_dir.join("frame_0.png"), frames_dir.join("frame_1.png"));
        if !frame0.exists() || !frame1.exists() {
            eprintln!("skipping: {} not present on this machine", frames_dir.display());
            return;
        }
        let Ok(api_key) = std::env::var("GEMINI_API_KEY") else {
            eprintln!("skipping: GEMINI_API_KEY not set");
            return;
        };

        let describer = vlm_describer::Describer::new(vlm_describer::Backend::Gemini(vlm_describer::Gemini::new(
            api_key,
            "gemini-3.5-flash",
        )));
        let synthesizer =
            kokoro_timestamped::KokoroSynthesizer::new(kokoro_timestamped::Voice::Name("af_heart".to_string()), "model", None, 1.0)
                .await
                .expect("load Kokoro");

        let sink: EventSink = Arc::new(|event: Event| println!("{event:?}"));
        let pipeline: Pipeline<vlm_describer::Describer, kokoro_timestamped::KokoroSynthesizer> = Pipeline::new(sink);
        pipeline.set_describer(Some(describer));
        pipeline.set_synthesizer(Some(synthesizer));

        pipeline.trigger(&Config::default(), image::open(&frame0).expect("decode frame_0.png"));
        std::thread::sleep(Duration::from_secs(5));
        pipeline.trigger(&Config::default(), image::open(&frame1).expect("decode frame_1.png"));
        std::thread::sleep(Duration::from_secs(10));
    }
}
