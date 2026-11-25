//! The pieces of one voice exchange, joined: the screenshot and what the user
//! said. They are produced independently and in either order, depending on
//! when the screenshot is taken (when the user starts speaking, or when they
//! finish), so this only waits until it has both. Every call carries the turn
//! number it was made for, and one for a turn that was superseded or aborted
//! is ignored.

use image::DynamicImage;

/// A finished turn: the screenshot and the transcript, ready to describe.
pub struct Ready {
    pub image: DynamicImage,
    pub text: String,
}

#[derive(Default)]
pub struct VoiceTurn {
    /// The newest turn's number; 0 before the first.
    current: u64,
    /// Whether that turn is still waiting for a part (or for the user).
    active: bool,
    frame: Option<DynamicImage>,
    transcript: Option<String>,
}

impl VoiceTurn {
    /// Starts a new turn, dropping whatever the previous one had gathered.
    pub fn begin(&mut self) -> u64 {
        self.current += 1;
        self.active = true;
        self.frame = None;
        self.transcript = None;
        self.current
    }

    /// Whether a turn is in progress. A regular (cycle) screenshot that
    /// arrives meanwhile is not wanted.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Whether `turn` is the one in progress.
    pub fn is_current(&self, turn: u64) -> bool {
        self.active && turn == self.current
    }

    /// The screenshot for `turn`. Returns the finished turn if the transcript
    /// was already there.
    pub fn set_frame(&mut self, turn: u64, image: DynamicImage) -> Option<Ready> {
        if !self.is_current(turn) {
            return None;
        }
        self.frame = Some(image);
        self.take_if_ready()
    }

    /// What the user said in `turn`. Returns the finished turn if the
    /// screenshot was already there.
    pub fn set_transcript(&mut self, turn: u64, text: String) -> Option<Ready> {
        if !self.is_current(turn) {
            return None;
        }
        self.transcript = Some(text);
        self.take_if_ready()
    }

    /// Gives up `turn`, dropping what it gathered. Returns whether it was the
    /// turn in progress.
    pub fn abort(&mut self, turn: u64) -> bool {
        if !self.is_current(turn) {
            return false;
        }
        self.reset();
        true
    }

    /// Gives up the turn in progress, if any (the cycle stopped).
    pub fn reset(&mut self) {
        self.active = false;
        self.frame = None;
        self.transcript = None;
    }

    fn take_if_ready(&mut self) -> Option<Ready> {
        if self.frame.is_some() && self.transcript.is_some() {
            self.active = false;
            Some(Ready {
                image: self.frame.take().expect("checked above"),
                text: self.transcript.take().expect("checked above"),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::new(2, 2))
    }

    #[test]
    fn it_is_ready_when_the_screenshot_comes_first() {
        let mut turn = VoiceTurn::default();
        let n = turn.begin();
        assert!(turn.set_frame(n, image()).is_none());
        let ready = turn.set_transcript(n, "hello".to_string()).expect("both parts are there");
        assert_eq!(ready.text, "hello");
        assert!(!turn.is_active(), "a finished turn is no longer in progress");
    }

    #[test]
    fn it_is_ready_when_the_transcript_comes_first() {
        let mut turn = VoiceTurn::default();
        let n = turn.begin();
        assert!(turn.set_transcript(n, "hello".to_string()).is_none());
        let ready = turn.set_frame(n, image()).expect("both parts are there");
        assert_eq!(ready.text, "hello");
    }

    #[test]
    fn a_part_for_an_older_turn_is_ignored() {
        let mut turn = VoiceTurn::default();
        let old = turn.begin();
        let new = turn.begin();
        assert!(turn.set_frame(old, image()).is_none());
        // The newer turn didn't get the old screenshot.
        assert!(turn.set_transcript(new, "hello".to_string()).is_none());
        assert!(turn.is_active());
    }

    #[test]
    fn beginning_a_turn_drops_what_the_previous_one_had() {
        let mut turn = VoiceTurn::default();
        let old = turn.begin();
        turn.set_frame(old, image());
        let new = turn.begin();
        assert!(turn.set_transcript(new, "hello".to_string()).is_none(), "the old screenshot is gone");
    }

    #[test]
    fn an_aborted_turn_gathers_nothing_more() {
        let mut turn = VoiceTurn::default();
        let n = turn.begin();
        turn.set_frame(n, image());
        assert!(turn.abort(n));
        assert!(!turn.is_active());
        assert!(turn.set_transcript(n, "hello".to_string()).is_none());
        assert!(!turn.abort(n), "aborting twice does nothing");
    }

    #[test]
    fn a_reset_ends_the_turn_in_progress() {
        let mut turn = VoiceTurn::default();
        let n = turn.begin();
        turn.reset();
        assert!(!turn.is_active());
        assert!(turn.set_frame(n, image()).is_none());
    }

    #[test]
    fn nothing_is_in_progress_before_the_first_turn() {
        let turn = VoiceTurn::default();
        assert!(!turn.is_active());
        assert!(!turn.is_current(0));
    }
}
