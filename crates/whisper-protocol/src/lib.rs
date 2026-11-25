//! The conversation between the app and `whisper-helper`, the separate
//! program that runs whisper.cpp. Keeping whisper in its own process keeps its
//! copy of the ggml library apart from llama.cpp's, which are different
//! versions: in one process on Linux the dynamic linker mixes them up and
//! llama.cpp aborts.
//!
//! The helper answers requests on stdin with replies on stdout. Everything is
//! a frame: a 4-byte little-endian length and that many bytes of JSON. A
//! [`Request::Transcribe`] is followed by its audio, `samples` raw
//! little-endian `f32`s (16 kHz mono) and no frame header. The helper exits
//! when stdin is closed. Its stderr is for log lines.
//!
//! [`serve`] is the helper's side, over an [`Engine`]; [`Client`] is the app's.

use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize};

/// The largest JSON frame accepted; the real ones are tiny.
const MAX_FRAME: usize = 1 << 20;
/// The most audio a request may carry: ten minutes at 16 kHz. A larger count
/// is a corrupted stream, not a recording.
pub const MAX_SAMPLES: usize = 16_000 * 60 * 10;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Which GPUs there are, and their memory.
    Gpus,
    /// Loads the model file at `path`, onto GPU `gpu_device` (one of the
    /// `id`s from [`Response::Gpus`]) or the CPU.
    Load { path: String, gpu_device: Option<i32> },
    /// Transcribes `samples` samples of audio, which follow. `language` is a
    /// code such as `"en"`, or `None` to detect it.
    Transcribe { language: Option<String>, samples: usize },
    /// Frees the model (and with it the GPU memory).
    Unload,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Gpus { gpus: Vec<Gpu> },
    Loaded,
    Text { text: String },
    Unloaded,
    Error { message: String },
}

/// A GPU the helper can run on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gpu {
    /// What `Load::gpu_device` takes.
    pub id: i32,
    pub name: String,
    /// Bytes of video memory free now, and in all.
    pub free: u64,
    pub total: u64,
}

/// What the helper does for each request; whisper is behind this.
pub trait Engine {
    fn gpus(&mut self) -> Vec<Gpu>;
    fn load(&mut self, path: &str, gpu_device: Option<i32>) -> Result<(), String>;
    fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<String, String>;
    fn unload(&mut self);
}

/// Answers requests from `input` on `output` until `input` ends, which is a
/// normal end (`Ok`). A stream that breaks off in the middle of a request, or
/// holds nonsense, is an error.
pub fn serve<E: Engine>(engine: &mut E, input: &mut impl Read, output: &mut impl Write) -> io::Result<()> {
    while let Some(frame) = read_frame(input)? {
        let request: Request = serde_json::from_slice(&frame).map_err(invalid)?;
        let response = match request {
            Request::Gpus => Response::Gpus { gpus: engine.gpus() },
            Request::Load { path, gpu_device } => match engine.load(&path, gpu_device) {
                Ok(()) => Response::Loaded,
                Err(message) => Response::Error { message },
            },
            Request::Transcribe { language, samples } => {
                let audio = read_samples(input, samples)?;
                match engine.transcribe(&audio, language.as_deref()) {
                    Ok(text) => Response::Text { text },
                    Err(message) => Response::Error { message },
                }
            }
            Request::Unload => {
                engine.unload();
                Response::Unloaded
            }
        };
        write_frame(output, &serde_json::to_vec(&response).map_err(invalid)?)?;
        output.flush()?;
    }
    Ok(())
}

/// The app's side: sends a request and waits for the reply. Errors are
/// messages that can be shown, whether the helper said no or went away.
pub struct Client<R, W> {
    input: R,
    output: W,
}

impl<R: Read, W: Write> Client<R, W> {
    /// `input` is the helper's stdout, `output` its stdin.
    pub fn new(input: R, output: W) -> Self {
        Self { input, output }
    }

    pub fn gpus(&mut self) -> Result<Vec<Gpu>, String> {
        match self.ask(&Request::Gpus, &[])? {
            Response::Gpus { gpus } => Ok(gpus),
            other => unexpected(other),
        }
    }

    pub fn load(&mut self, path: &str, gpu_device: Option<i32>) -> Result<(), String> {
        match self.ask(&Request::Load { path: path.to_string(), gpu_device }, &[])? {
            Response::Loaded => Ok(()),
            other => unexpected(other),
        }
    }

    pub fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<String, String> {
        let request = Request::Transcribe { language: language.map(str::to_string), samples: samples.len() };
        match self.ask(&request, samples)? {
            Response::Text { text } => Ok(text),
            other => unexpected(other),
        }
    }

    pub fn unload(&mut self) -> Result<(), String> {
        match self.ask(&Request::Unload, &[])? {
            Response::Unloaded => Ok(()),
            other => unexpected(other),
        }
    }

    fn ask(&mut self, request: &Request, samples: &[f32]) -> Result<Response, String> {
        let gone = |e: io::Error| format!("the speech recognition helper stopped: {e}");
        let json = serde_json::to_vec(request).map_err(|e| e.to_string())?;
        write_frame(&mut self.output, &json).map_err(gone)?;
        write_samples(&mut self.output, samples).map_err(gone)?;
        self.output.flush().map_err(gone)?;
        let frame = read_frame(&mut self.input)
            .map_err(gone)?
            .ok_or_else(|| gone(io::Error::new(io::ErrorKind::UnexpectedEof, "it closed its output")))?;
        match serde_json::from_slice(&frame).map_err(|e| format!("unreadable reply from the speech recognition helper: {e}"))? {
            Response::Error { message } => Err(message),
            response => Ok(response),
        }
    }
}

fn unexpected<T>(response: Response) -> Result<T, String> {
    Err(format!("unexpected reply from the speech recognition helper: {response:?}"))
}

fn invalid(error: impl std::error::Error + Send + Sync + 'static) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

fn write_frame(output: &mut impl Write, json: &[u8]) -> io::Result<()> {
    output.write_all(&u32::try_from(json.len()).map_err(invalid)?.to_le_bytes())?;
    output.write_all(json)
}

/// The next frame, or `None` if the stream ended before one began.
fn read_frame(input: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut length = [0u8; 4];
    let mut have = 0;
    while have < length.len() {
        match input.read(&mut length[have..])? {
            0 if have == 0 => return Ok(None),
            0 => return Err(io::ErrorKind::UnexpectedEof.into()),
            n => have += n,
        }
    }
    let length = u32::from_le_bytes(length) as usize;
    if length > MAX_FRAME {
        return Err(invalid(io::Error::other(format!("a frame of {length} bytes"))));
    }
    let mut frame = vec![0; length];
    input.read_exact(&mut frame)?;
    Ok(Some(frame))
}

fn write_samples(output: &mut impl Write, samples: &[f32]) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    output.write_all(&bytes)
}

fn read_samples(input: &mut impl Read, count: usize) -> io::Result<Vec<f32>> {
    if count > MAX_SAMPLES {
        return Err(invalid(io::Error::other(format!("{count} samples of audio"))));
    }
    let mut bytes = vec![0; count * 4];
    input.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::pipe;

    /// Answers what it is asked, from what it was given.
    #[derive(Default)]
    struct Fake {
        loaded: Option<(String, Option<i32>)>,
    }

    impl Engine for Fake {
        fn gpus(&mut self) -> Vec<Gpu> {
            vec![Gpu { id: 1, name: "Fake GPU".to_string(), free: 3 << 30, total: 8 << 30 }]
        }

        fn load(&mut self, path: &str, gpu_device: Option<i32>) -> Result<(), String> {
            if path.ends_with("missing.bin") {
                return Err("no such model".to_string());
            }
            self.loaded = Some((path.to_string(), gpu_device));
            Ok(())
        }

        fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<String, String> {
            let (path, device) = self.loaded.clone().ok_or("no model is loaded")?;
            let sum = samples.iter().fold(0.0, |total, sample| total + sample);
            Ok(format!("{path} {device:?} {} samples, sum {sum}, language {language:?}", samples.len()))
        }

        fn unload(&mut self) {
            self.loaded = None;
        }
    }

    /// A client talking to `Fake` through real pipes, with `serve` on a thread.
    fn connect() -> (Client<std::io::PipeReader, std::io::PipeWriter>, std::thread::JoinHandle<io::Result<()>>) {
        let (mut server_input, client_output) = pipe().unwrap();
        let (client_input, mut server_output) = pipe().unwrap();
        let server = std::thread::spawn(move || serve(&mut Fake::default(), &mut server_input, &mut server_output));
        (Client::new(client_input, client_output), server)
    }

    #[test]
    fn a_whole_conversation() {
        let (mut client, server) = connect();
        assert_eq!(client.gpus().unwrap(), [Gpu { id: 1, name: "Fake GPU".to_string(), free: 3 << 30, total: 8 << 30 }]);
        client.load("model.bin", Some(1)).unwrap();
        assert_eq!(
            client.transcribe(&[0.5, 0.25, -1.0], Some("ja")).unwrap(),
            "model.bin Some(1) 3 samples, sum -0.25, language Some(\"ja\")"
        );
        assert_eq!(client.transcribe(&[], None).unwrap(), "model.bin Some(1) 0 samples, sum 0, language None");
        client.unload().unwrap();
        assert_eq!(client.transcribe(&[0.0], None).unwrap_err(), "no model is loaded");

        drop(client); // closing the helper's stdin ends it
        assert!(server.join().unwrap().is_ok());
    }

    #[test]
    fn what_the_engine_refuses_comes_back_as_a_message_and_the_conversation_goes_on() {
        let (mut client, _server) = connect();
        assert_eq!(client.load("missing.bin", None).unwrap_err(), "no such model");
        client.load("model.bin", None).unwrap();
        assert!(client.transcribe(&[1.0], None).is_ok());
    }

    #[test]
    fn half_a_minute_of_audio_goes_through_intact() {
        let (mut client, _server) = connect();
        client.load("m.bin", None).unwrap();
        let audio: Vec<f32> = (0..16_000 * 30).map(|i| (i % 7) as f32 * 0.125).collect();
        let sum = audio.iter().fold(0.0, |total, sample| total + sample);
        assert_eq!(
            client.transcribe(&audio, None).unwrap(),
            format!("m.bin None {} samples, sum {sum}, language None", audio.len())
        );
    }

    #[test]
    fn a_helper_that_is_gone_is_an_error_not_a_hang() {
        let (client_input, server_output) = pipe().unwrap();
        let (_server_input, client_output) = pipe().unwrap();
        drop(server_output); // the helper's stdout closed: it died
        let mut client = Client::new(client_input, client_output);
        let error = client.gpus().unwrap_err();
        assert!(error.contains("speech recognition helper stopped"), "{error}");
    }

    #[test]
    fn a_stream_that_breaks_off_in_a_request_is_an_error_and_a_clean_end_is_not() {
        let mut engine = Fake::default();
        let mut out = Vec::new();
        assert!(serve(&mut engine, &mut io::empty(), &mut out).is_ok(), "nothing at all is a clean end");

        let mut half_length: &[u8] = &[7, 0];
        assert_eq!(serve(&mut engine, &mut half_length, &mut out).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);

        let mut request = Vec::new();
        write_frame(&mut request, &serde_json::to_vec(&Request::Transcribe { language: None, samples: 100 }).unwrap()).unwrap();
        request.extend_from_slice(&[0; 40]); // only 10 of the 100 samples
        assert_eq!(serve(&mut engine, &mut request.as_slice(), &mut out).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn nonsense_is_refused_before_anything_big_is_allocated() {
        let mut engine = Fake::default();
        let mut out = Vec::new();

        let mut huge_frame: &[u8] = &u32::MAX.to_le_bytes();
        assert_eq!(serve(&mut engine, &mut huge_frame, &mut out).unwrap_err().kind(), io::ErrorKind::InvalidData);

        let mut not_json = Vec::new();
        write_frame(&mut not_json, b"hello").unwrap();
        assert_eq!(serve(&mut engine, &mut not_json.as_slice(), &mut out).unwrap_err().kind(), io::ErrorKind::InvalidData);

        let mut too_much_audio = Vec::new();
        let request = Request::Transcribe { language: None, samples: MAX_SAMPLES + 1 };
        write_frame(&mut too_much_audio, &serde_json::to_vec(&request).unwrap()).unwrap();
        assert_eq!(serve(&mut engine, &mut too_much_audio.as_slice(), &mut out).unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn the_messages_have_the_shape_documented_for_other_ends() {
        let json = |r: &Request| serde_json::to_string(r).unwrap();
        assert_eq!(json(&Request::Gpus), r#"{"type":"gpus"}"#);
        assert_eq!(json(&Request::Load { path: "m.bin".into(), gpu_device: None }), r#"{"type":"load","path":"m.bin","gpu_device":null}"#);
        assert_eq!(
            json(&Request::Transcribe { language: Some("en".into()), samples: 5 }),
            r#"{"type":"transcribe","language":"en","samples":5}"#
        );
        assert_eq!(serde_json::to_string(&Response::Text { text: "hi".into() }).unwrap(), r#"{"type":"text","text":"hi"}"#);
    }
}
