//! Tests against a mock HTTP server, checking the requests the describer
//! sends and how it handles the responses.

use image::DynamicImage;
use serde_json::{Value, json};
use vlm_describer::{Backend, Describer, Error, Gemini, OpenAi, Prompts};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn image() -> DynamicImage {
    DynamicImage::new_rgb8(2, 2)
}

fn openai_reply(text: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .set_body_json(json!({ "choices": [{ "message": { "content": text } }] }))
}

async fn openai_server(reply: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(openai_reply(reply))
        .mount(&server)
        .await;
    server
}

fn openai_describer(server: &MockServer) -> Describer {
    let backend = OpenAi::new(None, "test-model").with_base_url(format!("{}/v1", server.uri()));
    Describer::new(Backend::OpenAi(backend))
}

/// The JSON bodies of all requests the server received, in order.
async fn bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|r| serde_json::from_slice(&r.body).unwrap())
        .collect()
}

/// The text of an OpenAI user message, without the images.
fn user_text(body: &Value) -> String {
    let user = body["messages"].as_array().unwrap().last().unwrap();
    user["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|part| part["text"].as_str())
        .collect()
}

#[tokio::test]
async fn openai_request_follows_the_placeholders() {
    let server = openai_server("A cat.").await;
    let mut describer = openai_describer(&server);

    let comment = describer.describe(&image(), Some(&image())).await.unwrap();
    assert_eq!(comment, "A cat.");

    let body = &bodies(&server).await[0];
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["max_tokens"], 300);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"][0]["text"], Prompts::default().system_prompt);
    // "Current image:\n<image>\n\nPrevious image:\n<image>\n\n..."
    let kinds: Vec<&str> = messages[1]["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|part| part["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["text", "image_url", "text", "image_url", "text"]);
    let url = messages[1]["content"][1]["image_url"]["url"].as_str().unwrap();
    assert!(url.starts_with("data:image/png;base64,"));
}

#[tokio::test]
async fn history_is_included_and_compacted() {
    let server = openai_server("comment").await;
    let mut describer = openai_describer(&server);
    describer.max_history_size = 2;
    describer.prompts.history_prompt = "HISTORY".into();
    describer.prompts.compact_prompt = "COMPACT".into();
    describer.prompts.prompt = "NEXT".into();
    describer.prompts.first_prompt = "FIRST".into();

    describer.describe(&image(), None).await.unwrap();
    describer.describe(&image(), Some(&image())).await.unwrap();
    // The history now has 2 comments, the maximum, so it is summarized first.
    describer.describe(&image(), Some(&image())).await.unwrap();

    let texts: Vec<String> = bodies(&server).await.iter().map(user_text).collect();
    assert_eq!(
        texts,
        [
            "FIRST",
            "HISTORY\ncomment\n---\nNEXT",
            "COMPACT\ncomment\ncomment",
            "HISTORY\ncomment\n---\nNEXT",
        ]
    );
    // The summary, plus the newest comment.
    assert_eq!(describer.history(), ["comment", "comment"]);
}

#[tokio::test]
async fn compaction_keeps_the_newest_comments() {
    let server = openai_server("comment").await;
    let mut describer = openai_describer(&server);
    describer.max_history_size = 3;
    describer.min_history_size = 2;
    describer.prompts.compact_prompt = "COMPACT".into();

    for _ in 0..3 {
        describer.describe(&image(), Some(&image())).await.unwrap();
    }
    describer.describe(&image(), Some(&image())).await.unwrap();

    let texts: Vec<String> = bodies(&server).await.iter().map(user_text).collect();
    // Only the oldest comment is summarized; the 2 newest are kept.
    assert_eq!(texts[3], "COMPACT\ncomment");
    assert_eq!(describer.history().len(), 4);
}

#[tokio::test]
async fn response_regex_extracts_the_comment() {
    let server = openai_server("<|start|>assistant to=user<|message|>Hello\nthere").await;
    let mut describer = openai_describer(&server)
        .with_response_re(r"<\|start\|>assistant to=user<\|message\|>(.*?)$")
        .unwrap();
    let comment = describer.describe(&image(), None).await.unwrap();
    assert_eq!(comment, "Hello\nthere");
}

#[tokio::test]
async fn http_errors_are_reported() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401).set_body_string("bad key"))
        .mount(&server)
        .await;
    let mut describer = openai_describer(&server);
    match describer.describe(&image(), None).await {
        Err(Error::Api { status: 401, body, .. }) => assert_eq!(body, "bad key"),
        other => panic!("expected an API error, got {:?}", other.map(|_| ())),
    }
    assert!(describer.history().is_empty());
}

#[tokio::test]
async fn gemini_request_and_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-test:generateContent"))
        .and(header("x-goog-api-key", "secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{ "content": { "parts": [
                { "text": "thinking...", "thought": true },
                { "text": "A " },
                { "text": "dog." },
            ] } }]
        })))
        .mount(&server)
        .await;
    let backend = Gemini::new("secret", "gemini-test").with_base_url(server.uri());
    let mut describer = Describer::new(Backend::Gemini(backend));

    // Thoughts are left out of the comment.
    assert_eq!(describer.describe(&image(), None).await.unwrap(), "A dog.");

    let body = &bodies(&server).await[0];
    assert_eq!(
        body["systemInstruction"]["parts"][0]["text"],
        Prompts::default().system_prompt
    );
    let parts = body["contents"][0]["parts"].as_array().unwrap();
    // "Current image:\n<image>\n\nDescribe..."
    assert_eq!(parts[0]["text"], "Current image:\n");
    assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
    assert!(parts[2]["text"].as_str().unwrap().starts_with("\n\nDescribe"));
}

#[tokio::test]
async fn generate_uses_no_images_or_history() {
    let server = openai_server("Hello!").await;
    let describer = openai_describer(&server);
    let greeting = describer.generate("Say hello.", Some("Be brief.")).await.unwrap();
    assert_eq!(greeting, "Hello!");

    let body = &bodies(&server).await[0];
    assert_eq!(body["messages"][0]["content"][0]["text"], "Be brief.");
    assert_eq!(body["messages"][1]["content"], json!([{ "type": "text", "text": "Say hello." }]));
    assert!(describer.history().is_empty());
}
