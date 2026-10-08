//! Google's Gemini API, through its REST interface (Python: `gemini.py`,
//! which uses the google-genai SDK).

use image::DynamicImage;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::backend::{png_base64, send};
use crate::error::{Error, Result};
use crate::prompts::{Segment, segments};

const PROVIDER: &str = "Gemini";
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

pub struct Gemini {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl Gemini {
    /// `model` is a model name like `gemini-3.5-flash`.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Sends requests somewhere other than Google, e.g. a test server.
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
            Ok(json!({ "inlineData": { "mimeType": "image/png", "data": png_base64(image)? } }))
        };

        // Without placeholders, the text comes first and then the images.
        let segments = segments(user_prompt);
        let mut parts = Vec::new();
        if !segments.iter().any(|s| matches!(s, Segment::Image(_))) {
            parts.push(json!({ "text": user_prompt }));
            for image in images {
                parts.push(image_part(image)?);
            }
        } else {
            for segment in segments {
                match segment {
                    Segment::Text(text) => parts.push(json!({ "text": text })),
                    Segment::Image(i) => {
                        if let Some(image) = images.get(i) {
                            parts.push(image_part(image)?);
                        }
                    }
                }
            }
        }

        let mut body = json!({ "contents": [{ "role": "user", "parts": parts }] });
        if let Some(system_prompt) = system_prompt.filter(|s| !s.is_empty()) {
            body["systemInstruction"] = json!({ "parts": [{ "text": system_prompt }] });
        }

        let url = format!("{}/v1beta/models/{}:generateContent", self.base_url, self.model);
        let request = self
            .client
            .post(url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body);
        let response: Response = send(PROVIDER, request).await?;

        // Like the SDK's `response.text`: the first candidate's text parts,
        // without the model's "thoughts".
        let text: String = response
            .candidates
            .into_iter()
            .next()
            .and_then(|c| c.content)
            .map(|content| {
                content
                    .parts
                    .into_iter()
                    .filter(|p| !p.thought)
                    .filter_map(|p| p.text)
                    .collect()
            })
            .unwrap_or_default();
        if text.is_empty() {
            return Err(Error::EmptyResponse { provider: PROVIDER });
        }
        Ok(text)
    }
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Option<Content>,
}

#[derive(Deserialize)]
struct Content {
    #[serde(default)]
    parts: Vec<Part>,
}

#[derive(Deserialize)]
struct Part {
    text: Option<String>,
    #[serde(default)]
    thought: bool,
}
