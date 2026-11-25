//! whisper.cpp behind the protocol's [`Engine`]: choosing the GPU, loading a
//! model, transcribing, and tidying what comes out.

use whisper_protocol::{Engine, Gpu};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// A segment whisper thinks is more likely silence than speech is dropped
/// when its probability of that is above this. Whisper makes up text such as
/// "Thank you." from noise.
const NO_SPEECH_MAX: f32 = 0.6;
/// Anything shorter than this (16 kHz samples) is not worth transcribing.
const MIN_SAMPLES: usize = 16_000 / 2;

pub struct WhisperEngine {
    context: Option<WhisperContext>,
}

impl WhisperEngine {
    pub fn new() -> Self {
        // whisper.cpp's and ggml's own output goes to the log, and so to stderr.
        whisper_rs::install_logging_hooks();
        Self { context: None }
    }

    /// Transcribes a second of silence, for the sake of the time it takes. On
    /// a GPU the driver compiles its shaders the first time they are used
    /// (about 25 s for the base model on an RTX 3060 Ti, kept by the driver
    /// for later runs); this way the first real recording doesn't wait for
    /// that. What it returns doesn't matter.
    fn warm_up(&mut self) {
        let started = std::time::Instant::now();
        let _ = self.transcribe(&[0.0; 16_000], Some("en"));
        log::info!("warmed up in {:.1}s", started.elapsed().as_secs_f64());
    }
}

impl Engine for WhisperEngine {
    fn gpus(&mut self) -> Vec<Gpu> {
        gpus()
    }

    fn load(&mut self, path: &str, gpu_device: Option<i32>) -> Result<(), String> {
        // The old model goes first, so that its memory is free for the new one.
        self.context = None;
        let mut params = WhisperContextParameters::default();
        params.use_gpu(gpu_device.is_some());
        if let Some(device) = gpu_device {
            params.gpu_device(device);
        }
        let context = WhisperContext::new_with_params(path, params).map_err(|e| format!("cannot load {path}: {e}"))?;
        self.context = Some(context);
        if gpu_device.is_some() {
            self.warm_up();
        }
        Ok(())
    }

    fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<String, String> {
        let context = self.context.as_ref().ok_or("No speech recognition model is loaded")?;
        if samples.len() < MIN_SAMPLES {
            return Ok(String::new());
        }
        let mut state = context.create_state().map_err(|e| e.to_string())?;
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

    fn unload(&mut self) {
        self.context = None;
    }
}

/// The GPUs whisper could use in this build, with their free memory.
#[cfg(feature = "vulkan")]
fn gpus() -> Vec<Gpu> {
    if !vulkan_loader_present() {
        return Vec::new();
    }
    whisper_rs::vulkan::list_devices()
        .into_iter()
        .filter(|device| !is_software_renderer(&device.name))
        .map(|device| Gpu { id: device.id, name: device.name, free: device.vram.free as u64, total: device.vram.total as u64 })
        .collect()
}

/// Apple's GPU shares the memory with the CPU, so there is no separate pool to
/// run out of: its memory is reported as unlimited.
#[cfg(all(feature = "metal", not(feature = "vulkan")))]
fn gpus() -> Vec<Gpu> {
    vec![Gpu { id: 0, name: "Apple GPU (Metal)".to_string(), free: u64::MAX, total: u64::MAX }]
}

/// A build without a GPU backend has none to offer.
#[cfg(not(any(feature = "vulkan", feature = "metal")))]
fn gpus() -> Vec<Gpu> {
    Vec::new()
}

/// Graphics cards that are really the CPU, drawing in software.
#[cfg(any(feature = "vulkan", test))]
fn is_software_renderer(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    ["llvmpipe", "swiftshader", "software", "basic render"].iter().any(|s| name.contains(s))
}

/// Whether the Vulkan loader is installed. The executable delay-loads it (see
/// build.rs), so on a machine without it, asking for the GPUs would crash.
#[cfg(all(feature = "vulkan", windows))]
fn vulkan_loader_present() -> bool {
    // SAFETY: loading a system library by name.
    unsafe { windows_sys::Win32::System::LibraryLoader::LoadLibraryA(b"vulkan-1.dll\0".as_ptr()) != std::ptr::null_mut() }
}

#[cfg(all(feature = "vulkan", not(windows)))]
fn vulkan_loader_present() -> bool {
    true
}

/// Takes the annotations whisper adds for sounds, such as `[BLANK_AUDIO]`,
/// `(music)` and ♪, out of `text`, and tidies the spaces. A bracket with no
/// partner is kept as it is.
fn clean_transcript(text: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn software_renderers_are_not_gpus() {
        for name in ["llvmpipe (LLVM 17.0.6, 256 bits)", "SwiftShader Device (Subzero)", "Microsoft Basic Render Driver"] {
            assert!(is_software_renderer(name), "{name}");
        }
        for name in ["NVIDIA GeForce RTX 3060 Ti", "AMD Radeon RX 7800 XT", "Intel(R) Arc(TM) A770 Graphics"] {
            assert!(!is_software_renderer(name), "{name}");
        }
    }

    #[test]
    fn without_a_model_it_says_so() {
        let mut engine = WhisperEngine { context: None };
        assert!(engine.transcribe(&[0.0; 16_000], None).unwrap_err().contains("No speech recognition model"));
        engine.unload(); // harmless
    }

    /// Loads a real whisper model and transcribes a 16 kHz mono 16-bit WAV
    /// with it, printing the text and the time. Needs both files; the sample
    /// is whisper.cpp's `samples/jfk.wav`, and a model from
    /// `ggerganov/whisper.cpp` (the app's cache has them once it has used
    /// voice input). Run on request:
    /// `ASR_MODEL=ggml-base-q5_1.bin ASR_WAV=jfk.wav cargo test -p whisper-helper --release -- --ignored --nocapture real_whisper`
    /// `ASR_GPU=0` (a device number, see `gpus()`) runs it on that GPU; it
    /// needs `--features vulkan`.
    #[test]
    #[ignore]
    fn real_whisper_transcribes_a_wav() {
        let model = std::env::var("ASR_MODEL").expect("ASR_MODEL: path of a whisper ggml model");
        let wav = std::env::var("ASR_WAV").expect("ASR_WAV: path of a 16 kHz mono WAV");
        let gpu_device = std::env::var("ASR_GPU").ok().map(|v| v.parse::<i32>().expect("ASR_GPU: a device number"));
        let bytes = std::fs::read(wav).expect("the WAV should be readable");
        // A canonical 44-byte header, then 16-bit samples.
        assert_eq!(&bytes[..4], b"RIFF");
        let samples: Vec<f32> = bytes[44..].chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect();

        let mut engine = WhisperEngine::new();
        for gpu in engine.gpus() {
            eprintln!("GPU {}: {} ({} MB free of {} MB)", gpu.id, gpu.name, gpu.free >> 20, gpu.total >> 20);
        }
        let started = std::time::Instant::now();
        engine.load(&model, gpu_device).expect("the model should load");
        eprintln!("on {}, loaded in {:.2}s", gpu_device.map_or("the CPU".to_string(), |d| format!("GPU {d}")), started.elapsed().as_secs_f64());
        let started = std::time::Instant::now();
        let text = engine.transcribe(&samples, None).expect("transcription");
        eprintln!("TRANSCRIPT ({:.2}s for {:.1}s of audio): {text}", started.elapsed().as_secs_f64(), samples.len() as f64 / 16_000.0);
        assert!(text.to_lowercase().contains("country"), "unexpected: {text}");
    }
}
