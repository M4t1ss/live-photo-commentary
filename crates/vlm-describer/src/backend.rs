//! The model a describer sends its prompts to.

use std::future::Future;
use std::time::Duration;

use base64::Engine;
use image::DynamicImage;

use crate::describer::CancelFn;
use crate::error::{Error, Result};
use crate::gemini::Gemini;
#[cfg(feature = "llama-cpp")]
use crate::llama_cpp::LlamaCpp;
use crate::openai::OpenAi;

/// Where prompts are sent.
pub enum Backend {
    /// Google's Gemini API.
    Gemini(Gemini),
    /// OpenAI's API, or any server with an OpenAI-compatible one, such as
    /// Ollama, LM Studio or llama.cpp's `llama-server`.
    OpenAi(OpenAi),
    /// A local GGUF model run by llama.cpp.
    #[cfg(feature = "llama-cpp")]
    LlamaCpp(LlamaCpp),
}

impl Backend {
    /// Returns [`Error::Cancelled`] as soon as `cancel` returns true.
    pub(crate) async fn prompt_model(
        &self,
        user_prompt: &str,
        images: &[&DynamicImage],
        system_prompt: Option<&str>,
        cancel: CancelFn<'_>,
    ) -> Result<String> {
        match self {
            Self::Gemini(gemini) => {
                until_cancelled(cancel, gemini.prompt_model(user_prompt, images, system_prompt)).await
            }
            Self::OpenAi(openai) => {
                until_cancelled(cancel, openai.prompt_model(user_prompt, images, system_prompt)).await
            }
            // Checks `cancel` itself, before every token.
            #[cfg(feature = "llama-cpp")]
            Self::LlamaCpp(model) => model.prompt_model(user_prompt, images, system_prompt, cancel).await,
        }
    }
}

/// How often a cloud request checks whether it has been cancelled.
const CANCEL_POLL: Duration = Duration::from_millis(100);

/// Runs `request` until it finishes or `cancel` returns true. Dropping the
/// request's future aborts the HTTP request.
async fn until_cancelled<T>(cancel: CancelFn<'_>, request: impl Future<Output = Result<T>>) -> Result<T> {
    tokio::pin!(request);
    loop {
        if cancel() {
            return Err(Error::Cancelled);
        }
        tokio::select! {
            result = &mut request => return result,
            () = tokio::time::sleep(CANCEL_POLL) => {}
        }
    }
}

/// An image as base64-encoded PNG.
pub(crate) fn png_base64(image: &DynamicImage) -> Result<String> {
    let mut png = std::io::Cursor::new(Vec::new());
    image.write_to(&mut png, image::ImageFormat::Png)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(png.into_inner()))
}

/// Sends a JSON request and returns the parsed response, turning HTTP errors
/// into [`Error::Api`].
pub(crate) async fn send<T: serde::de::DeserializeOwned>(
    provider: &'static str,
    request: reqwest::RequestBuilder,
) -> Result<T> {
    let request_error = |source| Error::Request { provider, source };
    let response = request.send().await.map_err(request_error)?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(Error::Api { provider, status: status.as_u16(), body });
    }
    response.json().await.map_err(request_error)
}
