//! Speech recognition: whisper.cpp in the separate program `whisper-helper`
//! (see `whisper-protocol` for why it is not in this process), asked from a
//! thread of its own because transcribing blocks for a second or more.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;

use whisper_protocol::Client;

use super::gpu::{self, Placement, UseGpu};

/// What the voice input needs from a speech recognizer. Implemented for
/// [`HelperAsr`]; tests implement it for fakes.
pub trait AsrOps: Send {
    /// What was said in `samples` (16 kHz mono). `language` is a code such as
    /// `"en"`; `None` detects it. Empty if there was no speech.
    fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<String, String>;
}

/// A running `whisper-helper`, and what it has loaded. If it dies (a GPU
/// driver crash, say), the next request starts it again and loads the model
/// again.
pub struct HelperAsr {
    helper: PathBuf,
    process: Process,
    /// The model and device last loaded, to load again after a restart.
    loaded: Option<(PathBuf, Option<i32>)>,
}

struct Process {
    child: Child,
    client: Client<BufReader<ChildStdout>, ChildStdin>,
}

impl Process {
    fn start(helper: &Path) -> Result<Self, String> {
        let mut command = Command::new(helper);
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(windows)]
        {
            // Don't open a console window when started from a GUI program.
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command.spawn().map_err(|e| format!("cannot start {}: {e}", helper.display()))?;
        let stdin = child.stdin.take().expect("stdin should be piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout should be piped"));
        // Its log lines (and whisper.cpp's) go to ours. Unread, they would fill
        // the pipe and stop it.
        let stderr = child.stderr.take().expect("stderr should be piped");
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                log::info!(target: "whisper_helper", "{line}");
            }
        });
        Ok(Self { child, client: Client::new(stdout, stdin) })
    }

    /// Whether the program has ended.
    fn exited(&mut self) -> bool {
        !matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // Closing its stdin would end it too, but a model being loaded or a
        // transcription in progress would hold that up.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl HelperAsr {
    pub fn start(helper: &Path) -> Result<Self, String> {
        Ok(Self { helper: helper.to_path_buf(), process: Process::start(helper)?, loaded: None })
    }

    /// The GPUs the helper can use, with their free memory.
    pub fn gpus(&mut self) -> Result<Vec<whisper_protocol::Gpu>, String> {
        self.process.client.gpus()
    }

    /// Loads the model file onto GPU `gpu_device`, or the CPU.
    pub fn load(&mut self, model: &Path, gpu_device: Option<i32>) -> Result<(), String> {
        let path = model.to_str().ok_or_else(|| format!("{} is not valid UTF-8", model.display()))?;
        self.process.client.load(path, gpu_device)?;
        self.loaded = Some((model.to_path_buf(), gpu_device));
        Ok(())
    }

    /// A new process, with the model loaded again.
    fn restart(&mut self) -> Result<(), String> {
        log::warn!("the speech recognition helper stopped; starting it again");
        self.process = Process::start(&self.helper)?;
        if let Some((model, device)) = self.loaded.clone() {
            self.load(&model, device)?;
        }
        Ok(())
    }
}

impl AsrOps for HelperAsr {
    fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<String, String> {
        match self.process.client.transcribe(samples, language) {
            // If the program has ended, that is why the request failed: start
            // it again and ask once more. If it is still there, the error is
            // its own answer.
            Err(_) if self.process.exited() => {
                self.restart()?;
                self.process.client.transcribe(samples, language)
            }
            result => result,
        }
    }
}

/// Where the helper program is: `WHISPER_HELPER` (its path), next to the
/// running program, or one folder up from it (for tests and examples, which
/// Cargo puts in a subfolder).
pub fn helper_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("WHISPER_HELPER") {
        return Some(path.into());
    }
    // e.g. `whisper-helper.exe` on Windows.
    let name = format!("whisper-helper{}", std::env::consts::EXE_SUFFIX);
    let exe = std::env::current_exe().ok()?;
    exe.ancestors().skip(1).take(2).map(|dir| dir.join(&name)).find(|path| path.is_file())
}

/// Starts the helper and loads `model` (`model_bytes` big) where `setting`
/// and the machine say: it asks the helper which GPUs it has, with their free
/// memory, and `gpu::place` decides. If a GPU turns out not to work, or the
/// helper dies when asked about them, the model goes on the CPU, and the
/// placement says why.
pub fn start_placed(
    helper: &Path,
    model: &Path,
    model_bytes: u64,
    setting: UseGpu,
    llama_devices: &[vlm_describer::BackendDevice],
) -> Result<(HelperAsr, Placement), String> {
    let mut asr = HelperAsr::start(helper)?;
    let mut gpus = Vec::new();
    let mut note = None;
    if setting != UseGpu::Cpu {
        match asr.gpus() {
            Ok(found) => gpus = gpu::gpu_infos(&found, llama_devices),
            Err(e) => {
                // A driver that crashes when asked is no GPU to use.
                log::warn!("could not ask the speech recognition helper for the GPUs: {e}");
                note = Some(format!("the GPUs could not be listed: {e}"));
                asr = HelperAsr::start(helper)?;
            }
        }
    }
    let mut placement = gpu::place(setting, model_bytes, &gpus);
    if let (Placement::Cpu { reason }, Some(note)) = (&mut placement, note) {
        *reason = note;
    }

    if let Placement::Gpu { device, name } = &placement {
        match asr.load(model, Some(*device)) {
            Ok(()) => return Ok((asr, placement)),
            Err(e) => {
                log::warn!("speech recognition cannot use {name}: {e}; using the CPU");
                // A failed load may have left the helper half-way; start afresh.
                asr = HelperAsr::start(helper)?;
                placement = Placement::Cpu { reason: format!("could not use {name}: {e}") };
            }
        }
    }
    asr.load(model, None)?;
    Ok((asr, placement))
}

enum AsrJob {
    SetModel(Option<Box<dyn AsrOps>>, mpsc::Sender<()>),
    Transcribe {
        samples: Vec<f32>,
        language: Option<String>,
        wanted: Box<dyn Fn() -> bool + Send>,
        done: Box<dyn FnOnce(Result<String, String>) + Send>,
    },
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
                    AsrJob::Transcribe { samples, language, wanted, done } => {
                        let result = if !wanted() {
                            Err("Not wanted any more".to_string())
                        } else {
                            match model.as_mut() {
                                Some(model) => model.transcribe(&samples, language.as_deref()),
                                None => Err("No speech recognition model is loaded".to_string()),
                            }
                        };
                        done(result);
                    }
                }
            }
        });
        Self { tx }
    }

    /// Replaces the recognizer (`None` frees it; for a [`HelperAsr`] that ends
    /// the helper program and so frees its GPU memory). Waits until any
    /// transcription in progress has finished, so call it from a thread that
    /// may block.
    pub fn set_model(&self, model: Option<Box<dyn AsrOps>>) {
        let (ack, rx) = mpsc::channel();
        if self.tx.send(AsrJob::SetModel(model, ack)).is_ok() {
            let _ = rx.recv();
        }
    }

    /// Transcribes `samples` (16 kHz mono) and calls `done` on the worker
    /// thread with the text, or the error. The jobs run one at a time; a job
    /// whose turn comes when `wanted` says no is not run, and `done` gets an
    /// error (a transcription in progress can't be stopped, but the ones
    /// waiting behind it can be skipped).
    pub fn transcribe(
        &self,
        samples: Vec<f32>,
        language: Option<String>,
        wanted: impl Fn() -> bool + Send + 'static,
        done: impl FnOnce(Result<String, String>) + Send + 'static,
    ) {
        if let Err(mpsc::SendError(AsrJob::Transcribe { done, .. })) = self.tx.send(AsrJob::Transcribe {
            samples,
            language,
            wanted: Box::new(wanted),
            done: Box::new(done),
        }) {
            done(Err("The speech recognition thread has stopped".to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Says what it was given, and that it was.
    struct Echo;

    impl AsrOps for Echo {
        fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<String, String> {
            Ok(format!("{} samples in {}", samples.len(), language.unwrap_or("auto")))
        }
    }

    fn transcribe(worker: &AsrWorker, samples: usize, language: Option<&str>) -> Result<String, String> {
        let (tx, rx) = mpsc::channel();
        worker.transcribe(vec![0.0; samples], language.map(str::to_string), || true, move |result| {
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

    /// Holds up its first transcription until told to go on.
    struct Slow(Option<mpsc::Receiver<()>>);

    impl AsrOps for Slow {
        fn transcribe(&mut self, samples: &[f32], _language: Option<&str>) -> Result<String, String> {
            if let Some(gate) = self.0.take() {
                let _ = gate.recv();
            }
            Ok(format!("{} samples", samples.len()))
        }
    }

    #[test]
    fn a_job_that_is_not_wanted_when_its_turn_comes_is_skipped() {
        let (go, gate) = mpsc::channel();
        let worker = AsrWorker::new();
        worker.set_model(Some(Box::new(Slow(Some(gate)))));
        let wanted = Arc::new(AtomicBool::new(true));
        let (tx, rx) = mpsc::channel();
        let ask = |n: usize, wanted: Arc<AtomicBool>| {
            let tx = tx.clone();
            worker.transcribe(vec![0.0; n], None, move || wanted.load(Ordering::SeqCst), move |result| {
                let _ = tx.send((n, result));
            });
        };
        ask(1, Arc::new(AtomicBool::new(true))); // being transcribed
        ask(2, Arc::clone(&wanted)); // waiting behind it
        ask(3, Arc::new(AtomicBool::new(true)));
        wanted.store(false, Ordering::SeqCst); // the second is superseded meanwhile
        go.send(()).unwrap();

        let answers: Vec<_> = (0..3).map(|_| rx.recv_timeout(Duration::from_secs(5)).unwrap()).collect();
        assert_eq!(answers[0], (1, Ok("1 samples".to_string())));
        assert!(answers[1].1.is_err(), "the superseded job is not run");
        assert_eq!(answers[2], (3, Ok("3 samples".to_string())));
    }

    #[test]
    fn requests_are_answered_in_order() {
        let worker = AsrWorker::new();
        worker.set_model(Some(Box::new(Echo)));
        let (tx, rx) = mpsc::channel();
        for n in 1..=5usize {
            let tx = tx.clone();
            worker.transcribe(vec![0.0; n], None, || true, move |result| {
                let _ = tx.send(result.unwrap());
            });
        }
        let answers: Vec<String> = (0..5).map(|_| rx.recv_timeout(Duration::from_secs(5)).unwrap()).collect();
        assert_eq!(answers, (1..=5).map(|n| format!("{n} samples in auto")).collect::<Vec<_>>());
    }

    #[test]
    fn a_helper_that_is_not_there_is_an_error_that_says_which() {
        let error = HelperAsr::start(Path::new("no/such/whisper-helper")).err().expect("it cannot start");
        assert!(error.contains("cannot start") && error.contains("whisper-helper"), "{error}");
    }

    /// Runs the real helper program (build it first: `cargo build -p
    /// whisper-helper`, with `--features vulkan` for the GPU) with a real
    /// model and a 16 kHz mono 16-bit WAV, as the app does: placed by
    /// `start_placed`, then asked to transcribe, then killed and asked again,
    /// which must start it afresh. Run on request:
    /// `ASR_MODEL=ggml-base-q5_1.bin ASR_WAV=jfk.wav cargo test -p app --lib --features dynamic-backends,vulkan -- --ignored --nocapture real_helper`
    /// `ASR_USE_GPU=cpu|gpu|auto` (default `auto`) is the setting.
    #[test]
    #[ignore]
    fn real_helper_transcribes_a_wav_and_survives_being_killed() {
        let helper = helper_path().expect("whisper-helper is not built (cargo build -p whisper-helper)");
        let model = PathBuf::from(std::env::var("ASR_MODEL").expect("ASR_MODEL: path of a whisper ggml model"));
        let wav = std::env::var("ASR_WAV").expect("ASR_WAV: path of a 16 kHz mono WAV");
        let setting = UseGpu::from_config(&std::env::var("ASR_USE_GPU").unwrap_or_default());
        let bytes = std::fs::read(wav).expect("the WAV should be readable");
        let samples: Vec<f32> = bytes[44..].chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect();

        vlm_describer::load_backends(None);
        let started = std::time::Instant::now();
        let (mut asr, placement) =
            start_placed(&helper, &model, std::fs::metadata(&model).unwrap().len(), setting, &vlm_describer::backend_devices())
                .expect("the helper should start and load the model");
        eprintln!("placed {placement:?} in {:.2}s", started.elapsed().as_secs_f64());

        for round in ["first", "after the kill"] {
            if round == "after the kill" {
                asr.process.child.kill().unwrap();
                let _ = asr.process.child.wait();
            }
            let started = std::time::Instant::now();
            let text = asr.transcribe(&samples, None).expect("transcription");
            eprintln!("{round}: TRANSCRIPT ({:.2}s for {:.1}s of audio): {text}", started.elapsed().as_secs_f64(), samples.len() as f64 / 16_000.0);
            assert!(text.to_lowercase().contains("country"), "unexpected: {text}");
        }
    }
}
