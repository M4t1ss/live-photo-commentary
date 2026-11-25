//! whisper.cpp as a separate program for the app's voice input; see
//! `whisper-protocol` for why, and for what it is asked. Reads requests on
//! stdin, answers on stdout (nothing else may write there), logs to stderr,
//! and exits when stdin is closed.

mod logger;
mod whisper;

fn main() {
    logger::init();
    let mut engine = whisper::WhisperEngine::new();
    let result = whisper_protocol::serve(&mut engine, &mut std::io::stdin().lock(), &mut std::io::stdout().lock());
    if let Err(e) = result {
        log::error!("the conversation with the app broke off: {e}");
        std::process::exit(1);
    }
}
