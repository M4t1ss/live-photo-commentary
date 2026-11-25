//! The voice input controller: from the hotkey, the microphone's level and
//! the cycle's state, decides when a voice turn starts, when the recording
//! ends, and what happens to it. A plain state machine: [`Input`]s go in, and
//! what it does comes out as calls on [`Effects`], so that all of it can be
//! tested without a microphone, a keyboard or a model.
//!
//! Two ways to start a turn, chosen by [`Activation`]:
//! - push to talk: a press of the hotkey, ended by the release, by another
//!   press, or by silence ([`EndMode`]); the microphone is opened per turn;
//! - voice threshold: while the cycle runs, the microphone stays open and a
//!   turn starts when the level stays above the threshold; it ends on silence.

use crate::config::Config;

use super::vad::{Detection, FRAME_MS, SpeechDetector};

/// A push-to-talk recording that ends on silence gives up if nothing was said
/// within this long.
const NO_SPEECH_TIMEOUT_FRAMES: usize = 8_000 / FRAME_MS;
/// The longest a recording is allowed to go on.
const MAX_RECORDING_FRAMES: usize = super::audio::MAX_RECORDING_S * 1000 / FRAME_MS;
/// A recording shorter than this (16 kHz samples) is dropped without asking
/// for a transcript: a tap on the key, not speech.
const MIN_SAMPLES: usize = 16_000 * 3 / 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Off,
    PushToTalk,
    Threshold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndMode {
    Release,
    Toggle,
    Silence,
}

/// When the screenshot for a turn is taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMoment {
    /// When the user starts speaking.
    Start,
    /// When the user is done speaking.
    End,
}

/// The settings the controller works from.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub activation: Activation,
    pub end: EndMode,
    pub capture: CaptureMoment,
    pub threshold_db: f32,
    /// Whether the user's voice may start a turn while the avatar is speaking.
    pub barge_in: bool,
}

impl Settings {
    /// From the config; a value it doesn't know is the default one.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            activation: match cfg.voice_activation.as_str() {
                "push_to_talk" => Activation::PushToTalk,
                "threshold" => Activation::Threshold,
                _ => Activation::Off,
            },
            end: match cfg.voice_input_end.as_str() {
                "toggle" => EndMode::Toggle,
                "silence" => EndMode::Silence,
                _ => EndMode::Release,
            },
            capture: if cfg.voice_capture_moment == "end" { CaptureMoment::End } else { CaptureMoment::Start },
            threshold_db: cfg.voice_threshold_db,
            barge_in: cfg.voice_barge_in,
        }
    }
}

/// What can happen to the controller.
#[derive(Debug)]
pub enum Input {
    Settings(Settings),
    /// The commentary cycle started or stopped. Voice input only works while
    /// it runs.
    CycleStarted,
    CycleStopped,
    KeyPressed,
    KeyReleased,
    /// The level in dBFS of the next `FRAME_MS` of the microphone.
    Level(f32),
    /// The avatar started or stopped speaking (reported by the frontend).
    Speaking(bool),
    /// The settings screen wants the microphone's level (for its meter).
    Monitor(bool),
    /// The transcription of `turn` finished.
    Transcribed { turn: u64, text: Result<String, String> },
}

/// What the controller does to the world.
pub trait Effects {
    /// Starts capturing; the levels come back as [`Input::Level`].
    fn open_mic(&mut self) -> Result<(), String>;
    fn close_mic(&mut self);
    /// Starts keeping everything captured, from just before now.
    fn start_recording(&mut self);
    /// Ends the recording and returns it as 16 kHz mono.
    fn finish_recording(&mut self) -> Result<Vec<f32>, String>;
    /// Ends the recording and throws it away.
    fn discard_recording(&mut self);
    /// Starts a voice turn, interrupting the avatar; returns its number.
    fn begin_turn(&mut self) -> u64;
    /// Asks for the screenshot for `turn`.
    fn request_capture(&mut self, turn: u64);
    /// Tells that the recording ended and is being transcribed.
    fn transcribing(&mut self, turn: u64);
    /// Starts transcribing; the result comes back as [`Input::Transcribed`].
    fn transcribe(&mut self, turn: u64, samples: Vec<f32>);
    fn set_transcript(&mut self, turn: u64, text: String);
    /// Ends `turn` without an answer.
    fn abort_turn(&mut self, turn: u64);
    /// A level for the settings screen's meter (only while it asked for it).
    fn level(&mut self, db: f32);
    fn error(&mut self, message: String);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Via {
    Key,
    Voice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Idle,
    Listening { turn: u64, via: Via },
    Transcribing { turn: u64 },
}

pub struct Controller<E: Effects> {
    fx: E,
    settings: Settings,
    running: bool,
    speaking: bool,
    monitor: bool,
    /// Whether the hotkey is held; the plugin repeats `pressed` while it is.
    key_down: bool,
    mic_open: bool,
    detector: SpeechDetector,
    state: State,
    /// How many frames the current recording has lasted.
    frames: usize,
}

impl<E: Effects> Controller<E> {
    pub fn new(fx: E, settings: Settings) -> Self {
        let detector = SpeechDetector::new(settings.threshold_db);
        Self {
            fx,
            settings,
            running: false,
            speaking: false,
            monitor: false,
            key_down: false,
            mic_open: false,
            detector,
            state: State::Idle,
            frames: 0,
        }
    }

    pub fn handle(&mut self, input: Input) {
        match input {
            Input::Settings(settings) => {
                self.detector.set_threshold(settings.threshold_db);
                self.settings = settings;
                if self.settings.activation == Activation::Off {
                    self.cancel();
                }
                if self.state == State::Idle {
                    self.detector.reset();
                }
                self.sync_mic();
            }
            Input::CycleStarted => {
                self.running = true;
                self.detector.reset();
                self.sync_mic();
            }
            Input::CycleStopped => {
                self.running = false;
                self.cancel();
                self.sync_mic();
            }
            Input::KeyPressed => {
                if !self.key_down {
                    self.key_down = true;
                    self.key_pressed();
                }
            }
            Input::KeyReleased => {
                if self.key_down {
                    self.key_down = false;
                    self.key_released();
                }
            }
            Input::Level(db) => {
                if self.monitor {
                    self.fx.level(db);
                }
                self.level(db);
            }
            Input::Speaking(speaking) => self.speaking = speaking,
            Input::Monitor(on) => {
                self.monitor = on;
                self.sync_mic();
            }
            Input::Transcribed { turn, text } => self.transcribed(turn, text),
        }
    }

    /// Opens or closes the microphone to match what needs it. Returns
    /// whether it is open.
    fn sync_mic(&mut self) -> bool {
        let needed = self.monitor
            || (self.running && self.settings.activation == Activation::Threshold)
            || matches!(self.state, State::Listening { .. });
        if needed && !self.mic_open {
            match self.fx.open_mic() {
                Ok(()) => self.mic_open = true,
                Err(message) => self.fx.error(message),
            }
        } else if !needed && self.mic_open {
            self.fx.close_mic();
            self.mic_open = false;
        }
        self.mic_open
    }

    /// Drops whatever turn is going on.
    fn cancel(&mut self) {
        match self.state {
            State::Listening { turn, .. } => {
                self.fx.discard_recording();
                self.fx.abort_turn(turn);
            }
            State::Transcribing { turn } => self.fx.abort_turn(turn),
            State::Idle => {}
        }
        self.state = State::Idle;
    }

    fn key_pressed(&mut self) {
        if !self.running || self.settings.activation != Activation::PushToTalk {
            return;
        }
        match self.state {
            State::Idle | State::Transcribing { .. } => self.start_turn(Via::Key),
            State::Listening { turn, via: Via::Key } if self.settings.end == EndMode::Toggle => self.finish(turn),
            // Ending by release or by silence: a press while recording is nothing.
            State::Listening { .. } => {}
        }
    }

    fn key_released(&mut self) {
        if let State::Listening { turn, via: Via::Key } = self.state {
            if self.settings.end == EndMode::Release {
                self.finish(turn);
            }
        }
    }

    fn level(&mut self, db: f32) {
        match self.state {
            // Also while a turn is being transcribed: new speech supersedes it.
            State::Idle | State::Transcribing { .. }
                if self.running && self.settings.activation == Activation::Threshold =>
            {
                if self.detector.push(db) == Detection::SpeechStarted {
                    if self.speaking && !self.settings.barge_in {
                        // The avatar's own voice, or other sound: not a turn.
                        self.detector.reset();
                    } else {
                        self.start_turn(Via::Voice);
                    }
                }
            }
            State::Listening { turn, via } => {
                self.frames += 1;
                let ended = self.detector.push(db) == Detection::SpeechEnded;
                let ends_on_silence = via == Via::Voice || self.settings.end == EndMode::Silence;
                if ends_on_silence && ended {
                    self.finish(turn);
                } else if via == Via::Key
                    && self.settings.end == EndMode::Silence
                    && !self.detector.speaking()
                    && self.frames >= NO_SPEECH_TIMEOUT_FRAMES
                {
                    self.give_up(turn);
                } else if self.frames >= MAX_RECORDING_FRAMES {
                    self.finish(turn);
                }
            }
            _ => {}
        }
    }

    /// Starts a voice turn. Whatever was going on is interrupted, including a
    /// turn that is still being transcribed.
    fn start_turn(&mut self, via: Via) {
        let turn = self.fx.begin_turn();
        self.state = State::Listening { turn, via };
        self.frames = 0;
        if via == Via::Key {
            self.detector.reset();
        }
        if !self.sync_mic() {
            self.fx.abort_turn(turn);
            self.state = State::Idle;
            return;
        }
        self.fx.start_recording();
        if self.settings.capture == CaptureMoment::Start {
            self.fx.request_capture(turn);
        }
    }

    /// Ends the recording of `turn` and has it transcribed.
    fn finish(&mut self, turn: u64) {
        self.detector.reset();
        let samples = match self.fx.finish_recording() {
            Ok(samples) => samples,
            Err(message) => {
                self.fx.error(message);
                return self.give_up(turn);
            }
        };
        if samples.len() < MIN_SAMPLES {
            return self.give_up(turn);
        }
        if self.settings.capture == CaptureMoment::End {
            self.fx.request_capture(turn);
        }
        self.fx.transcribing(turn);
        self.state = State::Transcribing { turn };
        self.sync_mic();
        self.fx.transcribe(turn, samples);
    }

    /// Ends `turn` without an answer.
    fn give_up(&mut self, turn: u64) {
        self.fx.discard_recording();
        self.fx.abort_turn(turn);
        self.state = State::Idle;
        self.detector.reset();
        self.sync_mic();
    }

    fn transcribed(&mut self, turn: u64, text: Result<String, String>) {
        if self.state != (State::Transcribing { turn }) {
            return; // superseded by a newer turn, or the cycle stopped
        }
        self.state = State::Idle;
        match text {
            Ok(text) => self.fx.set_transcript(turn, text),
            Err(message) => {
                self.fx.error(message);
                self.fx.abort_turn(turn);
            }
        }
        self.detector.reset();
        self.sync_mic();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records what the controller did, as short strings.
    #[derive(Default)]
    struct Fake {
        calls: Vec<String>,
        turns: u64,
        mic_fails: bool,
        /// What `finish_recording` returns: this many samples.
        recording_len: usize,
    }

    impl Fake {
        fn new() -> Self {
            Self { recording_len: 16_000, ..Self::default() }
        }
    }

    impl Effects for Fake {
        fn open_mic(&mut self) -> Result<(), String> {
            if self.mic_fails {
                self.calls.push("open_mic failed".into());
                return Err("no microphone".into());
            }
            self.calls.push("open_mic".into());
            Ok(())
        }
        fn close_mic(&mut self) {
            self.calls.push("close_mic".into());
        }
        fn start_recording(&mut self) {
            self.calls.push("start_recording".into());
        }
        fn finish_recording(&mut self) -> Result<Vec<f32>, String> {
            self.calls.push("finish_recording".into());
            Ok(vec![0.0; self.recording_len])
        }
        fn discard_recording(&mut self) {
            self.calls.push("discard_recording".into());
        }
        fn begin_turn(&mut self) -> u64 {
            self.turns += 1;
            self.calls.push(format!("begin_turn {}", self.turns));
            self.turns
        }
        fn request_capture(&mut self, turn: u64) {
            self.calls.push(format!("capture {turn}"));
        }
        fn transcribing(&mut self, turn: u64) {
            self.calls.push(format!("transcribing {turn}"));
        }
        fn transcribe(&mut self, turn: u64, samples: Vec<f32>) {
            self.calls.push(format!("transcribe {turn} ({} samples)", samples.len()));
        }
        fn set_transcript(&mut self, turn: u64, text: String) {
            self.calls.push(format!("transcript {turn} {text:?}"));
        }
        fn abort_turn(&mut self, turn: u64) {
            self.calls.push(format!("abort {turn}"));
        }
        fn level(&mut self, db: f32) {
            self.calls.push(format!("level {db}"));
        }
        fn error(&mut self, message: String) {
            self.calls.push(format!("error {message}"));
        }
    }

    const LOUD: f32 = -20.0;
    const QUIET: f32 = -70.0;

    fn settings(activation: Activation, end: EndMode) -> Settings {
        Settings { activation, end, capture: CaptureMoment::Start, threshold_db: -45.0, barge_in: false }
    }

    /// A controller whose cycle is running.
    fn running(settings: Settings) -> Controller<Fake> {
        let mut c = Controller::new(Fake::new(), settings);
        c.handle(Input::CycleStarted);
        c
    }

    fn feed(c: &mut Controller<Fake>, level: f32, ms: usize) {
        for _ in 0..ms / FRAME_MS {
            c.handle(Input::Level(level));
        }
    }

    /// The calls so far, and forget them.
    fn calls(c: &mut Controller<Fake>) -> Vec<String> {
        std::mem::take(&mut c.fx.calls)
    }

    fn press(c: &mut Controller<Fake>) {
        c.handle(Input::KeyPressed);
    }

    fn release(c: &mut Controller<Fake>) {
        c.handle(Input::KeyReleased);
    }

    // ── Push to talk ──

    #[test]
    fn holding_the_key_records_and_releasing_it_transcribes() {
        let mut c = running(settings(Activation::PushToTalk, EndMode::Release));
        assert!(calls(&mut c).is_empty(), "push to talk does not keep the microphone open");

        press(&mut c);
        assert_eq!(calls(&mut c), ["begin_turn 1", "open_mic", "start_recording", "capture 1"]);
        feed(&mut c, LOUD, 1000);
        release(&mut c);
        assert_eq!(
            calls(&mut c),
            ["finish_recording", "transcribing 1", "close_mic", "transcribe 1 (16000 samples)"]
        );

        c.handle(Input::Transcribed { turn: 1, text: Ok("what is that?".into()) });
        assert_eq!(calls(&mut c), ["transcript 1 \"what is that?\""]);
    }

    #[test]
    fn the_plugin_repeats_press_and_release_while_the_key_is_held() {
        let mut c = running(settings(Activation::PushToTalk, EndMode::Release));
        press(&mut c);
        calls(&mut c);
        press(&mut c);
        press(&mut c);
        assert!(calls(&mut c).is_empty(), "repeats of the press do nothing");
        release(&mut c);
        assert_eq!(calls(&mut c).len(), 4);
        release(&mut c);
        release(&mut c);
        assert!(calls(&mut c).is_empty(), "repeats of the release do nothing");
    }

    #[test]
    fn in_toggle_mode_a_second_press_ends_the_recording_but_holding_the_first_does_not() {
        let mut c = running(settings(Activation::PushToTalk, EndMode::Toggle));
        press(&mut c);
        press(&mut c); // key repeat while held
        release(&mut c); // no effect in this mode
        feed(&mut c, LOUD, 1000);
        assert_eq!(calls(&mut c), ["begin_turn 1", "open_mic", "start_recording", "capture 1"]);

        press(&mut c);
        assert_eq!(
            calls(&mut c),
            ["finish_recording", "transcribing 1", "close_mic", "transcribe 1 (16000 samples)"]
        );
    }

    #[test]
    fn in_silence_mode_the_recording_ends_after_the_speech_is_over() {
        let mut c = running(settings(Activation::PushToTalk, EndMode::Silence));
        press(&mut c);
        release(&mut c); // nothing happens on release
        calls(&mut c);
        feed(&mut c, QUIET, 1000); // before the user starts
        feed(&mut c, LOUD, 1500);
        assert!(calls(&mut c).is_empty());
        feed(&mut c, QUIET, 1000);
        assert_eq!(
            calls(&mut c),
            ["finish_recording", "transcribing 1", "close_mic", "transcribe 1 (16000 samples)"]
        );
    }

    #[test]
    fn in_silence_mode_it_gives_up_if_nothing_is_said() {
        let mut c = running(settings(Activation::PushToTalk, EndMode::Silence));
        press(&mut c);
        calls(&mut c);
        feed(&mut c, QUIET, 7_000);
        assert!(calls(&mut c).is_empty());
        feed(&mut c, QUIET, 1_000);
        assert_eq!(calls(&mut c), ["discard_recording", "abort 1", "close_mic"]);
    }

    #[test]
    fn a_tap_that_is_too_short_to_be_speech_is_dropped() {
        let mut c = running(settings(Activation::PushToTalk, EndMode::Release));
        c.fx.recording_len = 1_000;
        press(&mut c);
        calls(&mut c);
        release(&mut c);
        assert_eq!(calls(&mut c), ["finish_recording", "discard_recording", "abort 1", "close_mic"]);
    }

    #[test]
    fn a_recording_ends_by_itself_at_the_longest_length() {
        let mut c = running(settings(Activation::PushToTalk, EndMode::Release));
        press(&mut c);
        calls(&mut c);
        feed(&mut c, LOUD, 29_000);
        assert!(calls(&mut c).is_empty());
        feed(&mut c, LOUD, 1_000);
        assert!(calls(&mut c).contains(&"transcribe 1 (16000 samples)".to_string()));
    }

    #[test]
    fn the_key_does_nothing_unless_the_cycle_runs_and_the_mode_is_push_to_talk() {
        let mut idle = Controller::new(Fake::new(), settings(Activation::PushToTalk, EndMode::Release));
        press(&mut idle);
        assert!(calls(&mut idle).is_empty(), "the cycle is not running");

        let mut off = running(settings(Activation::Off, EndMode::Release));
        press(&mut off);
        assert!(calls(&mut off).is_empty());

        let mut threshold = running(settings(Activation::Threshold, EndMode::Release));
        calls(&mut threshold);
        press(&mut threshold);
        assert!(calls(&mut threshold).is_empty(), "in threshold mode the hotkey is not used");
    }

    #[test]
    fn the_screenshot_is_taken_when_the_user_starts_or_when_they_finish() {
        let mut start = running(Settings { capture: CaptureMoment::Start, ..settings(Activation::PushToTalk, EndMode::Release) });
        press(&mut start);
        release(&mut start);
        let calls_start = calls(&mut start);
        assert_eq!(calls_start.iter().position(|c| c == "capture 1"), Some(3), "right after recording starts: {calls_start:?}");

        let mut end = running(Settings { capture: CaptureMoment::End, ..settings(Activation::PushToTalk, EndMode::Release) });
        press(&mut end);
        assert!(!calls(&mut end).contains(&"capture 1".to_string()), "not at the start");
        release(&mut end);
        assert_eq!(
            calls(&mut end),
            ["finish_recording", "capture 1", "transcribing 1", "close_mic", "transcribe 1 (16000 samples)"]
        );
    }

    #[test]
    fn a_press_while_transcribing_starts_a_new_turn_and_the_old_result_is_ignored() {
        let mut c = running(settings(Activation::PushToTalk, EndMode::Release));
        press(&mut c);
        release(&mut c);
        calls(&mut c);

        press(&mut c);
        assert_eq!(calls(&mut c), ["begin_turn 2", "open_mic", "start_recording", "capture 2"]);
        c.handle(Input::Transcribed { turn: 1, text: Ok("old".into()) });
        assert!(calls(&mut c).is_empty(), "turn 1 was superseded");
    }

    #[test]
    fn a_microphone_that_will_not_open_is_reported_and_the_turn_ends() {
        let mut c = running(settings(Activation::PushToTalk, EndMode::Release));
        c.fx.mic_fails = true;
        press(&mut c);
        assert_eq!(calls(&mut c), ["begin_turn 1", "open_mic failed", "error no microphone", "abort 1"]);
        release(&mut c);
        assert!(calls(&mut c).is_empty());

        c.fx.mic_fails = false;
        press(&mut c);
        assert!(calls(&mut c).contains(&"start_recording".to_string()), "it works again once the microphone does");
    }

    #[test]
    fn a_failed_transcription_is_reported_and_the_turn_ends() {
        let mut c = running(settings(Activation::PushToTalk, EndMode::Release));
        press(&mut c);
        release(&mut c);
        calls(&mut c);
        c.handle(Input::Transcribed { turn: 1, text: Err("No speech recognition model is loaded".into()) });
        assert_eq!(calls(&mut c), ["error No speech recognition model is loaded", "abort 1"]);
    }

    #[test]
    fn stopping_the_cycle_or_switching_voice_input_off_drops_the_turn() {
        for stop in [Input::CycleStopped, Input::Settings(settings(Activation::Off, EndMode::Release))] {
            let mut c = running(settings(Activation::PushToTalk, EndMode::Release));
            press(&mut c);
            calls(&mut c);
            c.handle(stop);
            assert_eq!(calls(&mut c), ["discard_recording", "abort 1", "close_mic"]);
        }
    }

    // ── Voice threshold ──

    #[test]
    fn in_threshold_mode_the_microphone_is_open_exactly_while_the_cycle_runs() {
        let mut c = Controller::new(Fake::new(), settings(Activation::Threshold, EndMode::Release));
        assert!(calls(&mut c).is_empty());
        c.handle(Input::CycleStarted);
        assert_eq!(calls(&mut c), ["open_mic"]);
        c.handle(Input::CycleStopped);
        assert_eq!(calls(&mut c), ["close_mic"]);
    }

    #[test]
    fn speaking_above_the_threshold_starts_a_turn_and_silence_ends_it() {
        let mut c = running(settings(Activation::Threshold, EndMode::Release));
        calls(&mut c);
        feed(&mut c, QUIET, 2000);
        assert!(calls(&mut c).is_empty());
        feed(&mut c, LOUD, 200);
        assert!(calls(&mut c).is_empty(), "a short burst is not speech");
        feed(&mut c, QUIET, 200);

        feed(&mut c, LOUD, 400);
        assert_eq!(calls(&mut c), ["begin_turn 1", "start_recording", "capture 1"]);
        feed(&mut c, LOUD, 1000);
        feed(&mut c, QUIET, 1000);
        assert_eq!(calls(&mut c), ["finish_recording", "transcribing 1", "transcribe 1 (16000 samples)"]);
        // The microphone stays open, waiting for the next time.
        c.handle(Input::Transcribed { turn: 1, text: Ok("hello".into()) });
        assert_eq!(calls(&mut c), ["transcript 1 \"hello\""]);
        feed(&mut c, LOUD, 400);
        assert_eq!(calls(&mut c)[0], "begin_turn 2");
    }

    #[test]
    fn while_the_avatar_speaks_the_voice_does_not_start_a_turn_unless_barge_in_is_on() {
        let mut c = running(settings(Activation::Threshold, EndMode::Release));
        calls(&mut c);
        c.handle(Input::Speaking(true));
        feed(&mut c, LOUD, 2000);
        assert!(calls(&mut c).is_empty(), "the avatar's own voice must not interrupt it");

        c.handle(Input::Speaking(false));
        feed(&mut c, LOUD, 400);
        assert_eq!(calls(&mut c)[0], "begin_turn 1");

        let mut barge = running(Settings { barge_in: true, ..settings(Activation::Threshold, EndMode::Release) });
        calls(&mut barge);
        barge.handle(Input::Speaking(true));
        feed(&mut barge, LOUD, 400);
        assert_eq!(calls(&mut barge)[0], "begin_turn 1", "with barge-in the voice interrupts speech");
    }

    #[test]
    fn a_new_onset_while_transcribing_starts_another_turn() {
        let mut c = running(settings(Activation::Threshold, EndMode::Release));
        feed(&mut c, LOUD, 400);
        feed(&mut c, QUIET, 1000);
        calls(&mut c);
        feed(&mut c, LOUD, 400);
        assert_eq!(calls(&mut c)[0], "begin_turn 2");
        c.handle(Input::Transcribed { turn: 1, text: Ok("old".into()) });
        assert!(calls(&mut c).is_empty());
    }

    #[test]
    fn changing_the_threshold_applies_at_once() {
        let mut c = running(Settings { threshold_db: -10.0, ..settings(Activation::Threshold, EndMode::Release) });
        calls(&mut c);
        feed(&mut c, LOUD, 2000); // -20 dB is below -10 dB
        assert!(calls(&mut c).is_empty());
        c.handle(Input::Settings(Settings { threshold_db: -30.0, ..settings(Activation::Threshold, EndMode::Release) }));
        feed(&mut c, LOUD, 400);
        assert_eq!(calls(&mut c)[0], "begin_turn 1");
    }

    // ── The settings screen's meter ──

    #[test]
    fn the_meter_gets_levels_only_while_asked_for_and_keeps_the_microphone_open_meanwhile() {
        let mut c = running(settings(Activation::PushToTalk, EndMode::Release));
        feed(&mut c, LOUD, 40);
        assert!(calls(&mut c).is_empty(), "nobody asked for levels");

        c.handle(Input::Monitor(true));
        assert_eq!(calls(&mut c), ["open_mic"]);
        c.handle(Input::Level(-33.0));
        assert_eq!(calls(&mut c), ["level -33"]);

        c.handle(Input::Monitor(false));
        assert_eq!(calls(&mut c), ["close_mic"]);
    }

    #[test]
    fn the_meter_does_not_close_a_microphone_that_a_recording_or_the_threshold_needs() {
        let mut c = running(settings(Activation::Threshold, EndMode::Release));
        calls(&mut c);
        c.handle(Input::Monitor(true));
        c.handle(Input::Monitor(false));
        assert!(calls(&mut c).is_empty(), "the threshold mode keeps it open");

        let mut ptt = running(settings(Activation::PushToTalk, EndMode::Release));
        ptt.handle(Input::Monitor(true));
        press(&mut ptt);
        calls(&mut ptt);
        ptt.handle(Input::Monitor(false));
        assert!(calls(&mut ptt).is_empty(), "a recording is going on");
    }

    #[test]
    fn settings_come_from_the_config_with_defaults_for_unknown_values() {
        let cfg = Config {
            voice_activation: "threshold".into(),
            voice_input_end: "toggle".into(),
            voice_capture_moment: "end".into(),
            voice_threshold_db: -33.0,
            voice_barge_in: true,
            ..Config::default()
        };
        assert_eq!(
            Settings::from_config(&cfg),
            Settings {
                activation: Activation::Threshold,
                end: EndMode::Toggle,
                capture: CaptureMoment::End,
                threshold_db: -33.0,
                barge_in: true
            }
        );
        let odd = Config { voice_activation: "??".into(), voice_input_end: "??".into(), voice_capture_moment: "??".into(), ..Config::default() };
        let settings = Settings::from_config(&odd);
        assert_eq!((settings.activation, settings.end, settings.capture), (Activation::Off, EndMode::Release, CaptureMoment::Start));
    }
}
