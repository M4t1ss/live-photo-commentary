//! OpenAI's chat completions API, or any OpenAI-compatible server (Python:
//! `openai.py`, which uses the openai SDK).

use image::DynamicImage;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::backend::{png_base64, send};
use crate::error::{Error, Result};
use crate::prompts::{Segment, segments};

const PROVIDER: &str = "OpenAI";
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

pub struct OpenAi {
    client: reqwest::Client,
    api_key: Option<String>,
    model: String,
    base_url: String,
}

impl OpenAi {
    /// `model` is a model name like `gpt-4o`. Local servers often need no
    /// API key.
    pub fn new(api_key: Option<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model: model.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Uses an OpenAI-compatible server instead, e.g.
    /// `http://localhost:11434/v1` for Ollama.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    pub(crate) async fn prompt_model(
        &self,
        user_prompt: &str,
        images: &[&DynamicImage],
        system_prompt: Option<&str>,
    ) -> Result<String> {
        let image_part = |image: &DynamicImage| -> Result<Value> {
            let url = format!("data:image/png;base64,{}", png_base64(image)?);
            Ok(json!({ "type": "image_url", "image_url": { "url": url } }))
        };
        let text_part = |text: &str| json!({ "type": "text", "text": text });

        // Without placeholders, the images come first and then the text.
        let segments = segments(user_prompt);
        let mut content = Vec::new();
        if !segments.iter().any(|s| matches!(s, Segment::Image(_))) {
            for image in images {
                content.push(image_part(image)?);
            }
            if !user_prompt.is_empty() {
                content.push(text_part(user_prompt));
            }
        } else {
            for segment in segments {
                match segment {
                    Segment::Text(text) => content.push(text_part(text)),
                    Segment::Image(i) => {
                        if let Some(image) = images.get(i) {
                            content.push(image_part(image)?);
                        }
                    }
                }
            }
        }

        let mut messages = Vec::new();
        if let Some(system_prompt) = system_prompt.filter(|s| !s.is_empty()) {
            messages.push(json!({ "role": "system", "content": [text_part(system_prompt)] }));
        }
        messages.push(json!({ "role": "user", "content": content }));

        let body = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7,
        });
        let request = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            // Like the Python version, which sends "none" without a key.
            .bearer_auth(self.api_key.as_deref().unwrap_or("none"))
            .json(&body);
        let response: Response = send(PROVIDER, request).await?;

        response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .filter(|text| !text.is_empty())
            .ok_or(Error::EmptyResponse { provider: PROVIDER })
    }
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    content: Option<String>,
}
