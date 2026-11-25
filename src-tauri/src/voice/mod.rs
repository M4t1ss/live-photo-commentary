//! Voice input: the microphone, deciding when the user speaks, and turning
//! the recording into text. The controller that ties them to the hotkey and
//! the pipeline's voice turns comes on top of this.

pub mod asr;
pub mod audio;
pub mod models;
pub mod vad;
