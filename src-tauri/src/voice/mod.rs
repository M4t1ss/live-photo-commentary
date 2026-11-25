//! Voice input: the microphone, deciding when the user speaks, and turning
//! the recording into text, and the controller that ties them to the hotkey
//! and the pipeline's voice turns.

pub mod asr;
pub mod audio;
pub mod controller;
pub mod gpu;
pub mod hotkey;
pub mod live;
pub mod models;
pub mod vad;
