//! Describes screenshots, like the app does frame by frame.
//!
//! Usage:
//!   cargo run --example describe -- gemini MODEL IMAGE...
//!   cargo run --example describe -- openai MODEL [--base-url URL] IMAGE...
//!
//! API keys come from the GEMINI_API_KEY or OPENAI_API_KEY environment
//! variables. With several images, each is described as the "current" one
//! with the image before it as "previous", keeping a history of 3 comments.

use vlm_describer::{Backend, Describer, Gemini, OpenAi};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        anyhow::bail!("usage: describe (gemini|openai) MODEL [--base-url URL] IMAGE...");
    }
    let provider = args.remove(0);
    let model = args.remove(0);
    let mut base_url = None;
    if args.first().is_some_and(|a| a == "--base-url") && args.len() >= 2 {
        base_url = Some(args.remove(1));
        args.remove(0);
    }

    let backend = match provider.as_str() {
        "gemini" => Backend::Gemini(Gemini::new(std::env::var("GEMINI_API_KEY")?, model)),
        "openai" => {
            let mut openai = OpenAi::new(std::env::var("OPENAI_API_KEY").ok(), model);
            if let Some(url) = base_url {
                openai = openai.with_base_url(url);
            }
            Backend::OpenAi(openai)
        }
        other => anyhow::bail!("unknown provider {other:?}"),
    };
    let mut describer = Describer::new(backend);
    describer.prompts = describer.prompts.with_tags(&["joy", "surprised", "sad"]);
    describer.max_history_size = 3;

    let mut previous = None;
    for path in &args {
        let image = image::open(path)?;
        let comment = describer.describe(&image, previous.as_ref()).await?;
        println!("{path}: {comment}\n");
        previous = Some(image);
    }
    Ok(())
}
