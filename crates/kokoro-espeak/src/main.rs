//! Phonemizes text with eSpeak NG for kokoro-timestamped.
//!
//! eSpeak NG is GPL, so it runs in this separate program rather than inside
//! kokoro-timestamped, which can then be permissively licensed.
//!
//! Protocol: one JSON object per line on stdin, `{"voice": "en-us", "text":
//! "..."}`, answered by one line on stdout, `{"phonemes": "..."}` or
//! `{"error": "..."}`. The program exits when stdin is closed.

mod espeak;

use std::io::{self, BufRead, Write};

use serde_json::{Value, json};

fn main() -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let response = match handle(&line?) {
            Ok(phonemes) => json!({ "phonemes": phonemes }),
            Err(error) => json!({ "error": error }),
        };
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}

fn handle(line: &str) -> Result<String, String> {
    let request: Value = serde_json::from_str(line).map_err(|e| format!("invalid request: {e}"))?;
    let field = |name| {
        request[name]
            .as_str()
            .ok_or_else(|| format!("request has no string {name:?}"))
    };
    espeak::phonemize(field("text")?, field("voice")?).map_err(|e| e.to_string())
}
