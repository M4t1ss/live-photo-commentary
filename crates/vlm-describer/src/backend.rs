//! The model a describer sends its prompts to.

use base64::Engine;
use image::DynamicImage;

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
    pub(crate) async fn prompt_model(
        &self,
        user_prompt: &str,
        images: &[&DynamicImage],
        system_prompt: Option<&str>,
    ) -> Result<String> {
        match self {
            Self::Gemini(gemini) => gemini.prompt_model(user_prompt, images, system_prompt).await,
            Self::OpenAi(openai) => openai.prompt_model(user_prompt, images, system_prompt).await,
            #[cfg(feature = "llama-cpp")]
            Self::LlamaCpp(model) => model.prompt_model(user_prompt, images, system_prompt).await,
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
