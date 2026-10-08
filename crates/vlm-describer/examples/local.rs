//! Describes screenshots with a local model, timing loading and each
//! description, like the app does frame by frame.
//!
//! Usage (release builds; debug builds are far too slow):
//!   cargo run --release --features llama-cpp --example local -- \
//!       llama-cpp ggml-org/Qwen3-VL-2B-Instruct-GGUF MODEL.gguf MMPROJ.gguf IMAGE...
//!
//! Set GPU_LAYERS to offload llama.cpp layers to the GPU (default 0, CPU only).

use std::time::Instant;

use vlm_describer::{Backend, Describer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        anyhow::bail!("usage: local llama-cpp REPO MODEL MMPROJ IMAGE...");
    }
    let engine = args.remove(0);

    let start = Instant::now();
    let backend = match engine.as_str() {
        #[cfg(feature = "llama-cpp")]
        "llama-cpp" => {
            let repo = args.remove(0);
            let model_file = args.remove(0);
            let mmproj_file = args.remove(0);
            let progress: vlm_describer::DownloadProgress =
                std::sync::Arc::new(|file, done, total| eprintln!("{file}: {done}/{total}"));
            let (model, mmproj) =
                vlm_describer::download_model(&repo, &model_file, &mmproj_file, Some(progress)).await?;
            println!("downloaded in {:.1}s", start.elapsed().as_secs_f64());
            let gpu_layers = std::env::var("GPU_LAYERS").map_or(Ok(0), |v| v.parse())?;
            Backend::LlamaCpp(vlm_describer::LlamaCpp::new(&model, &mmproj, gpu_layers)?)
        }
        other => anyhow::bail!("unknown or disabled engine {other:?}"),
    };
    println!("loaded in {:.1}s\n", start.elapsed().as_secs_f64());

    let mut describer = Describer::new(backend);
    describer.prompts = describer.prompts.with_tags(&["joy", "surprised", "sad"]);
    describer.max_history_size = 3;

    let mut previous = None;
    for path in &args {
        let image = image::open(path)?;
        let start = Instant::now();
        let comment = describer.describe(&image, previous.as_ref()).await?;
        println!("{path} ({:.1}s): {comment}\n", start.elapsed().as_secs_f64());
        previous = Some(image);
    }
    Ok(())
}
