//! Phonemization with eSpeak NG, which runs in the separate helper program
//! `kokoro-espeak` (in this workspace). eSpeak NG is GPL; keeping it in its
//! own process keeps this crate permissively licensed.
//!
//! The helper is started once and answers one JSON line per request; see
//! `kokoro-espeak/src/main.rs` for the protocol.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Mutex, PoisonError};

use serde_json::{Value, json};

use crate::error::{Error, Result};

/// A running helper process.
pub struct Espeak {
    helper: Mutex<Helper>,
}

struct Helper {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Espeak {
    /// Starts the helper, found through `KOKORO_ESPEAK` (its path), next to
    /// the running program, or one folder up from it (for tests and
    /// examples, which Cargo puts in a subfolder).
    pub fn start() -> Result<Self> {
        let path = helper_path().ok_or(Error::EspeakHelperNotFound)?;
        let mut command = Command::new(&path);
        command.stdin(Stdio::piped()).stdout(Stdio::piped());
        #[cfg(windows)]
        {
            // Don't open a console window when started from a GUI program.
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command.spawn().map_err(Error::EspeakHelper)?;
        let stdin = child.stdin.take().expect("stdin should be piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout should be piped"));
        Ok(Self { helper: Mutex::new(Helper { child, stdin, stdout }) })
    }

    /// Converts `text` to Kokoro phonemes with the eSpeak voice `voice` (e.g.
    /// `en-us`), keeping punctuation in place.
    pub fn phonemize(&self, text: &str, voice: &str) -> Result<String> {
        let mut helper = self.helper.lock().unwrap_or_else(PoisonError::into_inner);
        let request = json!({ "voice": voice, "text": text });
        writeln!(helper.stdin, "{request}").map_err(Error::EspeakHelper)?;
        helper.stdin.flush().map_err(Error::EspeakHelper)?;
        let mut line = String::new();
        if helper.stdout.read_line(&mut line).map_err(Error::EspeakHelper)? == 0 {
            let exited = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "helper exited");
            return Err(Error::EspeakHelper(exited));
        }
        let response: Value = serde_json::from_str(&line)
            .map_err(|e| Error::EspeakHelper(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        match (response["phonemes"].as_str(), response["error"].as_str()) {
            (Some(phonemes), _) => Ok(phonemes.to_string()),
            (None, Some(error)) => Err(Error::Espeak(error.to_string())),
            (None, None) => {
                let invalid = std::io::Error::new(std::io::ErrorKind::InvalidData, line);
                Err(Error::EspeakHelper(invalid))
            }
        }
    }
}

impl Drop for Helper {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn helper_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("KOKORO_ESPEAK") {
        return Some(path.into());
    }
    // e.g. `kokoro-espeak.exe` on Windows.
    let name = format!("kokoro-espeak{}", std::env::consts::EXE_SUFFIX);
    let exe = std::env::current_exe().ok()?;
    exe.ancestors()
        .skip(1)
        .take(2)
        .map(|dir| dir.join(&name))
        .find(|path| path.is_file())
}
