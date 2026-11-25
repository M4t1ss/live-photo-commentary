//! The microphone: choosing a device, capturing from it, keeping the audio
//! that was just before the speech started, and converting a recording to
//! what whisper takes (16 kHz mono).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use serde::Serialize;

use super::vad::{self, FRAME_MS};

/// The sample rate whisper takes.
pub const WHISPER_RATE: u32 = 16_000;
/// How much audio from before the speech started is kept, so the first
/// syllable isn't cut off.
pub const PRE_ROLL_MS: usize = 500;
/// The longest recording kept; older audio is dropped.
pub const MAX_RECORDING_S: usize = 30;

/// An input device, as the Settings list shows it.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct InputDevice {
    /// What `voice_input_device` stores: stable, unlike the name, which two
    /// devices can share.
    pub id: String,
    pub name: String,
    /// Whether this is the operating system's default input.
    pub is_default: bool,
}

/// The input devices, the default one first.
pub fn list_input_devices() -> Vec<InputDevice> {
    let host = cpal::default_host();
    let default_id = host.default_input_device().and_then(|d| d.id().ok()).map(|id| id.to_string());
    let mut devices: Vec<InputDevice> = host
        .input_devices()
        .into_iter()
        .flatten()
        .filter_map(|device| {
            let id = device.id().ok()?.to_string();
            let name = device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| id.clone());
            Some(InputDevice { is_default: default_id.as_deref() == Some(&id), id, name })
        })
        .collect();
    devices.sort_by_key(|d| !d.is_default);
    devices
}

/// The device with id `selected`, or the default input if none is selected or
/// that one is gone (unplugged), in which case the second value says so.
fn find_device(host: &cpal::Host, selected: Option<&str>) -> Result<(cpal::Device, Option<String>), String> {
    let mut warning = None;
    if let Some(selected) = selected {
        match selected.parse::<cpal::DeviceId>().ok().and_then(|id| host.device_by_id(&id)) {
            Some(device) => return Ok((device, None)),
            None => warning = Some(format!("Input device {selected:?} not found, using the default")),
        }
    }
    match host.default_input_device() {
        Some(device) => Ok((device, warning)),
        None => Err("No microphone found".to_string()),
    }
}

/// Mono from interleaved frames: the average of the channels.
pub fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved.chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() / channels as f32).collect()
}

/// A recording to 16 kHz, which is what whisper takes. `rate` is the rate
/// `samples` were captured at.
pub fn resample_to_16k(samples: &[f32], rate: u32) -> Result<Vec<f32>, String> {
    if rate == WHISPER_RATE || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let error = |e: &dyn std::fmt::Display| format!("cannot resample from {rate} Hz: {e}");
    let mut resampler =
        Fft::<f32>::new(rate as usize, WHISPER_RATE as usize, 1024, 1, FixedSync::Input).map_err(|e| error(&e))?;
    let input = InterleavedSlice::new(samples, 1, samples.len()).map_err(|e| error(&e))?;
    let output = resampler.process_all(&input, samples.len(), None).map_err(|e| error(&e))?;
    Ok(output.take_data())
}

/// What the microphone has delivered: measured in frames of `FRAME_MS`, and
/// kept, either just the last `PRE_ROLL_MS` while waiting for speech, or
/// everything since `start` while a recording is going.
pub struct Recorder {
    frame_len: usize,
    frame: Vec<f32>,
    pre_roll_len: usize,
    max_len: usize,
    samples: VecDeque<f32>,
    recording: bool,
}

impl Recorder {
    /// For mono audio at `rate` Hz.
    pub fn new(rate: u32) -> Self {
        let per_ms = rate as usize / 1000;
        Self {
            frame_len: per_ms * FRAME_MS,
            frame: Vec::new(),
            pre_roll_len: per_ms * PRE_ROLL_MS,
            max_len: rate as usize * MAX_RECORDING_S,
            samples: VecDeque::new(),
            recording: false,
        }
    }

    /// Takes the next mono samples and returns the level in dBFS of each
    /// frame they completed.
    pub fn push(&mut self, mono: &[f32]) -> Vec<f32> {
        self.samples.extend(mono);
        let keep = if self.recording { self.max_len } else { self.pre_roll_len };
        if self.samples.len() > keep {
            self.samples.drain(..self.samples.len() - keep);
        }

        let mut levels = Vec::new();
        let mut rest = mono;
        while !rest.is_empty() {
            let take = (self.frame_len - self.frame.len()).min(rest.len());
            self.frame.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.frame.len() == self.frame_len {
                levels.push(vad::level_db(&self.frame));
                self.frame.clear();
            }
        }
        levels
    }

    /// Starts keeping everything from now on, together with the pre-roll.
    pub fn start(&mut self) {
        self.recording = true;
    }

    #[cfg(test)]
    fn is_recording(&self) -> bool {
        self.recording
    }

    /// How many samples have been kept (the pre-roll included).
    #[cfg(test)]
    fn len(&self) -> usize {
        self.samples.len()
    }

    /// Ends the recording and returns it, starting over with an empty buffer.
    pub fn finish(&mut self) -> Vec<f32> {
        self.recording = false;
        self.frame.clear();
        std::mem::take(&mut self.samples).into()
    }
}

/// An open microphone stream, feeding a [`Recorder`]. Closes when dropped.
pub struct Mic {
    _stream: cpal::Stream,
    /// The rate of the audio in `recorder`.
    pub sample_rate: u32,
    #[allow(dead_code)] // used by the voice controller
    pub recorder: Arc<Mutex<Recorder>>,
    /// Set when the chosen device wasn't there and the default was used.
    pub warning: Option<String>,
}

impl Mic {
    /// Opens the device with id `device` (the default input if `None`, or if
    /// that one is gone) and starts capturing. `on_level` gets the level in
    /// dBFS of each `FRAME_MS`, on the audio thread, so it must be quick.
    pub fn open(device: Option<&str>, on_level: impl Fn(f32) + Send + 'static) -> Result<Mic, String> {
        let host = cpal::default_host();
        let (device, warning) = find_device(&host, device)?;
        let config = device.default_input_config().map_err(|e| format!("cannot use the microphone: {e}"))?;
        let sample_rate = config.sample_rate();
        let channels = config.channels() as usize;
        let recorder = Arc::new(Mutex::new(Recorder::new(sample_rate)));

        let stream = match config.sample_format() {
            SampleFormat::F32 => Self::build::<f32>(&device, config.into(), channels, &recorder, on_level),
            SampleFormat::I16 => Self::build::<i16>(&device, config.into(), channels, &recorder, on_level),
            SampleFormat::I32 => Self::build::<i32>(&device, config.into(), channels, &recorder, on_level),
            SampleFormat::U16 => Self::build::<u16>(&device, config.into(), channels, &recorder, on_level),
            other => return Err(format!("the microphone uses an unsupported sample format ({other})")),
        }
        .map_err(|e| format!("cannot open the microphone: {e}"))?;
        stream.play().map_err(|e| format!("cannot start the microphone: {e}"))?;
        Ok(Mic { _stream: stream, sample_rate, recorder, warning })
    }

    fn build<T>(
        device: &cpal::Device,
        config: cpal::StreamConfig,
        channels: usize,
        recorder: &Arc<Mutex<Recorder>>,
        on_level: impl Fn(f32) + Send + 'static,
    ) -> Result<cpal::Stream, cpal::Error>
    where
        T: SizedSample,
        f32: FromSample<T>,
    {
        let recorder = Arc::clone(recorder);
        device.build_input_stream(
            config,
            move |data: &[T], _| {
                let floats: Vec<f32> = data.iter().map(|&s| f32::from_sample(s)).collect();
                let mono = downmix(&floats, channels);
                let measured = match recorder.lock() {
                    Ok(mut recorder) => recorder.push(&mono),
                    Err(_) => return,
                };
                for db in measured {
                    on_level(db);
                }
            },
            |e| match e.kind() {
                // A late buffer, once per callback while the CPU is busy.
                cpal::ErrorKind::Xrun => log::debug!("microphone stream: {e}"),
                _ => log::warn!("microphone stream error: {e}"),
            },
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, hz: f32, seconds: f32) -> Vec<f32> {
        (0..(rate as f32 * seconds) as usize).map(|i| (i as f32 / rate as f32 * hz * std::f32::consts::TAU).sin() * 0.5).collect()
    }

    fn zero_crossings(samples: &[f32]) -> usize {
        samples.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count()
    }

    #[test]
    fn downmix_averages_the_channels() {
        assert_eq!(downmix(&[0.5, 1.0], 1), [0.5, 1.0]);
        assert_eq!(downmix(&[1.0, 0.0, 0.5, 0.5], 2), [0.5, 0.5]);
        assert_eq!(downmix(&[0.3, 0.3, 0.6, 1.0, 1.0, 1.0], 3), [0.4, 1.0]);
        assert_eq!(downmix(&[1.0, 0.0, 0.5], 2), [0.5], "a partial frame at the end is dropped");
    }

    #[test]
    fn resampling_keeps_the_pitch_and_changes_the_length() {
        for rate in [48_000, 44_100, 32_000] {
            let out = resample_to_16k(&sine(rate, 440.0, 1.0), rate).unwrap();
            assert!((out.len() as i64 - 16_000).abs() < 40, "{rate} Hz gave {} samples", out.len());
            // 440 Hz crosses zero 880 times a second.
            let crossings = zero_crossings(&out);
            assert!((crossings as i64 - 880).abs() < 30, "{rate} Hz gave {crossings} crossings");
        }
    }

    #[test]
    fn audio_that_is_already_16_khz_or_empty_is_returned_as_it_is() {
        let samples = sine(16_000, 440.0, 0.1);
        assert_eq!(resample_to_16k(&samples, 16_000).unwrap(), samples);
        assert!(resample_to_16k(&[], 48_000).unwrap().is_empty());
    }

    #[test]
    fn a_short_recording_can_be_resampled() {
        let out = resample_to_16k(&sine(48_000, 440.0, 0.05), 48_000).unwrap();
        assert!((out.len() as i64 - 800).abs() < 20, "{}", out.len());
    }

    #[test]
    fn the_recorder_measures_each_completed_frame_and_keeps_the_rest() {
        let mut recorder = Recorder::new(48_000); // a frame is 960 samples
        assert_eq!(recorder.push(&vec![0.1; 2500]).len(), 2);
        // 580 left over, 380 more complete the third frame.
        assert_eq!(recorder.push(&vec![0.1; 380]).len(), 1);
        let levels = recorder.push(&vec![0.1; 960]);
        assert_eq!(levels.len(), 1);
        assert!((levels[0] - (-20.0)).abs() < 0.1);
    }

    #[test]
    fn without_a_recording_only_the_pre_roll_is_kept() {
        let mut recorder = Recorder::new(16_000);
        recorder.push(&vec![0.1; 16_000 * 2]);
        assert_eq!(recorder.len(), 16_000 / 1000 * PRE_ROLL_MS);
    }

    #[test]
    fn a_recording_starts_with_the_pre_roll_and_keeps_everything_after() {
        let mut recorder = Recorder::new(16_000);
        recorder.push(&vec![0.1; 16_000]); // before the speech
        recorder.start();
        assert!(recorder.is_recording());
        recorder.push(&vec![0.2; 16_000]); // the speech

        let recording = recorder.finish();
        let pre_roll = 16_000 / 1000 * PRE_ROLL_MS;
        assert_eq!(recording.len(), pre_roll + 16_000);
        assert!(recording[..pre_roll].iter().all(|&s| s == 0.1));
        assert!(recording[pre_roll..].iter().all(|&s| s == 0.2));

        assert!(!recorder.is_recording());
        assert_eq!(recorder.len(), 0, "finishing starts over");
    }

    #[test]
    fn a_recording_is_capped_and_the_oldest_audio_is_dropped() {
        let mut recorder = Recorder::new(1_000);
        recorder.start();
        recorder.push(&vec![0.1; 1_000 * (MAX_RECORDING_S - 1)]);
        assert_eq!(recorder.len(), 1_000 * (MAX_RECORDING_S - 1));
        recorder.push(&vec![0.2; 1_000 * 5]);
        assert_eq!(recorder.len(), 1_000 * MAX_RECORDING_S);
        let recording = recorder.finish();
        assert_eq!(recording[0], 0.1);
        assert_eq!(*recording.last().unwrap(), 0.2);
    }

    /// Needs a microphone, and opens it for a moment (the system's recording
    /// indicator lights up): `cargo test -p app --lib --features
    /// dynamic-backends,vulkan -- --ignored --nocapture opens_the_default_microphone`
    #[test]
    #[ignore]
    fn opens_the_default_microphone() {
        for device in list_input_devices() {
            eprintln!("input device: {device:?}");
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let mic = Mic::open(None, move |db| {
            let _ = tx.send(db);
        })
        .expect("the default microphone should open");
        eprintln!("opened at {} Hz, warning: {:?}", mic.sample_rate, mic.warning);
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let levels: Vec<f32> = rx.try_iter().collect();
        eprintln!(
            "{} frames measured, loudest {:.1} dBFS, quietest {:.1} dBFS",
            levels.len(),
            levels.iter().cloned().fold(f32::MIN, f32::max),
            levels.iter().cloned().fold(f32::MAX, f32::min)
        );
        assert!(levels.len() > 50, "1.5 s should give about 75 frames, got {}", levels.len());
    }
}
