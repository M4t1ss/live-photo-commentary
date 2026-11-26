//! Deciding from the microphone's level when the user is speaking: a loudness
//! threshold, enough loud time within a short window to start, and a stretch
//! of quiet to end. It serves the "end on silence" mode of push-to-talk,
//! starting a turn by voice, and the level meter in Settings.

use std::collections::VecDeque;

/// How long one level measurement covers.
pub const FRAME_MS: usize = 20;
/// Speech starts when, within the last `ONSET_WINDOW_MS`, the level was above
/// the threshold for at least `ONSET_LOUD_MS` in all (not in a row: speech
/// dips below the threshold between syllables, and asking for an unbroken run
/// took seconds to get). That is enough to leave out a click or a few key
/// presses, and the louder stretch of a cough still counts.
pub const ONSET_WINDOW_MS: usize = 600;
pub const ONSET_LOUD_MS: usize = 320;
/// How long it must stay below the threshold for the speech to be over.
pub const HOLD_MS: usize = 1000;

/// What a measurement made of the speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detection {
    Nothing,
    SpeechStarted,
    SpeechEnded,
}

/// The loudness of `samples` in dBFS: 0 is the loudest a sample can be, and
/// silence is about -120.
pub fn level_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -120.0;
    }
    let mean_square = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    (10.0 * mean_square.max(1e-12).log10()).max(-120.0)
}

pub struct SpeechDetector {
    threshold_db: f32,
    speaking: bool,
    /// The last `ONSET_WINDOW_MS` of frames (before the speech started): whether each was loud.
    recent: VecDeque<bool>,
    /// How many of `recent` are loud.
    loud_frames: usize,
    /// Consecutive frames below the threshold (while speaking).
    quiet_frames: usize,
}

impl SpeechDetector {
    pub fn new(threshold_db: f32) -> Self {
        Self { threshold_db, speaking: false, recent: VecDeque::new(), loud_frames: 0, quiet_frames: 0 }
    }

    pub fn set_threshold(&mut self, threshold_db: f32) {
        self.threshold_db = threshold_db;
    }

    /// Whether speech has started and not yet ended.
    pub fn speaking(&self) -> bool {
        self.speaking
    }

    /// Back to waiting for speech.
    pub fn reset(&mut self) {
        self.speaking = false;
        self.recent.clear();
        self.loud_frames = 0;
        self.quiet_frames = 0;
    }

    /// Takes the level of the next frame (`FRAME_MS` long).
    pub fn push(&mut self, level_db: f32) -> Detection {
        let loud = level_db >= self.threshold_db;
        if self.speaking {
            self.quiet_frames = if loud { 0 } else { self.quiet_frames + 1 };
            if self.quiet_frames >= HOLD_MS / FRAME_MS {
                self.reset();
                return Detection::SpeechEnded;
            }
        } else {
            self.recent.push_back(loud);
            self.loud_frames += usize::from(loud);
            if self.recent.len() > ONSET_WINDOW_MS / FRAME_MS && self.recent.pop_front() == Some(true) {
                self.loud_frames -= 1;
            }
            if self.loud_frames >= ONSET_LOUD_MS.div_ceil(FRAME_MS) {
                self.reset();
                self.speaking = true;
                return Detection::SpeechStarted;
            }
        }
        Detection::Nothing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOUD: f32 = -20.0;
    const QUIET: f32 = -70.0;
    const THRESHOLD: f32 = -45.0;

    /// Feeds `ms` of `level` and returns the detections that were not `Nothing`.
    fn feed(detector: &mut SpeechDetector, level: f32, ms: usize) -> Vec<Detection> {
        (0..ms / FRAME_MS).map(|_| detector.push(level)).filter(|d| *d != Detection::Nothing).collect()
    }

    #[test]
    fn levels_are_in_dbfs() {
        assert_eq!(level_db(&[]), -120.0);
        assert_eq!(level_db(&[0.0; 100]), -120.0);
        assert!((level_db(&[1.0; 100]) - 0.0).abs() < 1e-4, "full scale is 0 dB");
        assert!((level_db(&[0.1; 100]) - (-20.0)).abs() < 1e-3);
        assert!((level_db(&[-0.1; 100]) - (-20.0)).abs() < 1e-3, "the sign does not matter");
        // A full-scale sine has an RMS of 1/sqrt(2), about -3 dB.
        let sine: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.05).sin()).collect();
        assert!((level_db(&sine) - (-3.0)).abs() < 0.1);
    }

    #[test]
    fn quiet_never_starts_speech() {
        let mut detector = SpeechDetector::new(THRESHOLD);
        assert!(feed(&mut detector, QUIET, 5000).is_empty());
        assert!(!detector.speaking());
    }

    #[test]
    fn a_short_burst_is_not_speech_but_a_long_one_is() {
        let mut detector = SpeechDetector::new(THRESHOLD);
        assert!(feed(&mut detector, LOUD, 100).is_empty());
        feed(&mut detector, QUIET, 700); // longer than the window: the burst is forgotten
        assert!(feed(&mut detector, LOUD, 100).is_empty(), "bursts far apart don't add up");
        feed(&mut detector, QUIET, 700);

        assert_eq!(feed(&mut detector, LOUD, 400), [Detection::SpeechStarted]);
        assert!(detector.speaking());
    }

    /// `loud` ms above the threshold, then `quiet` ms below it, over and over, for `total` ms; when
    /// (in ms from the start) the speech was detected, if it was.
    fn detected_at(pattern_loud: usize, pattern_quiet: usize, total: usize) -> Option<usize> {
        let mut detector = SpeechDetector::new(THRESHOLD);
        let mut at = 0;
        while at < total {
            let (level, ms) = if (at % (pattern_loud + pattern_quiet)) < pattern_loud { (LOUD, FRAME_MS) } else { (QUIET, FRAME_MS) };
            if detector.push(level) == Detection::SpeechStarted {
                return Some(at + ms);
            }
            at += FRAME_MS;
        }
        None
    }

    #[test]
    fn speech_that_dips_between_syllables_is_detected_within_half_a_second() {
        // About what the dumped recording of "Testing one two three" looked like: loud stretches of
        // 100-200 ms with gaps of 40-80 ms; an unbroken run of 250 ms took seconds to turn up.
        for (loud, quiet) in [(160, 60), (100, 60), (120, 80), (200, 40)] {
            let at = detected_at(loud, quiet, 3000).unwrap_or_else(|| panic!("{loud}/{quiet} ms was not detected"));
            assert!(at <= 600, "{loud} ms loud, {quiet} ms quiet: detected only after {at} ms");
        }
    }

    #[test]
    fn typing_and_clicks_are_not_speech() {
        // A key press is a loud spell of 40 ms, about ten a second at the fastest.
        assert_eq!(detected_at(40, 60, 5000), None, "fast typing");
        assert_eq!(detected_at(20, 480, 5000), None, "an occasional click");
        // A cough or a knock of 200 ms.
        assert_eq!(detected_at(200, 5000, 5000), None, "a short knock");
    }

    #[test]
    fn speech_ends_after_the_hold_off_of_quiet() {
        let mut detector = SpeechDetector::new(THRESHOLD);
        feed(&mut detector, LOUD, 400);
        assert!(feed(&mut detector, QUIET, 900).is_empty(), "not yet");
        assert_eq!(feed(&mut detector, QUIET, 100), [Detection::SpeechEnded]);
        assert!(!detector.speaking());
    }

    #[test]
    fn a_pause_shorter_than_the_hold_off_does_not_end_the_speech() {
        let mut detector = SpeechDetector::new(THRESHOLD);
        feed(&mut detector, LOUD, 400);
        assert!(feed(&mut detector, QUIET, 800).is_empty());
        assert!(feed(&mut detector, LOUD, 100).is_empty(), "speaking again restarts the wait");
        assert!(feed(&mut detector, QUIET, 800).is_empty());
        assert_eq!(feed(&mut detector, QUIET, 200), [Detection::SpeechEnded]);
    }

    #[test]
    fn after_the_end_it_waits_for_new_speech() {
        let mut detector = SpeechDetector::new(THRESHOLD);
        feed(&mut detector, LOUD, 400);
        feed(&mut detector, QUIET, 1000);
        assert_eq!(feed(&mut detector, LOUD, 400), [Detection::SpeechStarted]);
    }

    #[test]
    fn the_threshold_decides_what_is_loud_and_can_change() {
        let mut detector = SpeechDetector::new(-10.0);
        assert!(feed(&mut detector, LOUD, 1000).is_empty(), "-20 dB is below a -10 dB threshold");
        detector.set_threshold(-30.0);
        assert_eq!(feed(&mut detector, LOUD, 400), [Detection::SpeechStarted]);
    }

    #[test]
    fn a_reset_forgets_the_speech() {
        let mut detector = SpeechDetector::new(THRESHOLD);
        feed(&mut detector, LOUD, 400);
        detector.reset();
        assert!(!detector.speaking());
        assert!(feed(&mut detector, QUIET, 2000).is_empty(), "no end for speech that was forgotten");
    }
}
