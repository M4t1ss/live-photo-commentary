//! Local models with llama.cpp, which runs GGUF models. Experimental.

use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::Path;
use std::pin::Pin;

use image::DynamicImage;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker,
};
use llama_cpp_2::sampling::LlamaSampler;

use crate::error::{Error, Result};
use crate::prompts::{Segment, segments};

/// Room for two full-screen screenshots, the history and the reply.
const CONTEXT_SIZE: u32 = 8192;
const BATCH_SIZE: u32 = 2048;
/// Video memory kept free beside the model and its context, for the image
/// encoder's working memory (on top of the `mmproj` file's own size).
const IMAGE_ENCODER_RESERVE: usize = 1 << 30;
/// ggml's `GGML_LOG_LEVEL_WARN`, for the fitting output.
const FIT_LOG_LEVEL: i32 = 3;

/// How replies are generated. The defaults are the Python backend's
/// (`default_generation_args` in `local_describer.py`).
#[derive(Debug, Clone, PartialEq)]
pub struct Sampling {
    /// The longest reply, in tokens.
    pub max_new_tokens: usize,
    pub temperature: f32,
    /// Keep only the smallest set of tokens whose probabilities add up to
    /// this much. `None` (or 1.0) keeps all.
    pub top_p: Option<f32>,
    /// Keep only the most likely tokens. `None` keeps all.
    pub top_k: Option<i32>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { max_new_tokens: 200, temperature: 1.0, top_p: None, top_k: Some(50) }
    }
}

pub struct LlamaCpp {
    backend: LlamaBackend,
    model: LlamaModel,
    mtmd: MtmdContext,
    template: LlamaChatTemplate,
    /// How replies are generated.
    pub sampling: Sampling,
}

fn local_error(e: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Local(Box::new(e))
}

impl LlamaCpp {
    /// Loads a GGUF model and its vision projector (`mmproj`) file.
    /// `gpu_layers` is 0 for CPU only. Otherwise llama.cpp offloads as many
    /// layers as fit into the GPU's free memory (after the context and the
    /// image encoder), if it was built with a GPU backend that is loaded;
    /// a model that doesn't fit entirely runs partly on the CPU instead of
    /// overflowing into system memory, which is far slower.
    ///
    /// Only one can exist at a time: llama.cpp's backend is global, so drop
    /// the previous one before loading another.
    pub fn new(model_path: &Path, mmproj_path: &Path, gpu_layers: u32) -> Result<Self> {
        // llama.cpp logs a lot to stderr; send it to `tracing` instead.
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
        let backend = LlamaBackend::init().map_err(local_error)?;
        let mut params = LlamaModelParams::default();
        let mut use_gpu = gpu_layers > 0;
        if use_gpu {
            let devices = super::backend_devices();
            match super::preferred_gpu(&devices) {
                Some(gpu) => {
                    log::info!("using {} {}", gpu.backend, gpu.description);
                    params = params.with_devices(&[gpu.index]).map_err(local_error)?;
                }
                None => use_gpu = false,
            }
        }
        let mut params = Box::pin(params);
        if use_gpu {
            use_gpu = fit_to_memory(params.as_mut(), model_path, mmproj_path);
        }
        if !use_gpu {
            params = Box::pin(LlamaModelParams::default().with_n_gpu_layers(0));
        }
        let model = LlamaModel::load_from_file(&backend, model_path, &params).map_err(local_error)?;
        log::info!("model loaded with {} GPU layers", params.n_gpu_layers());
        let mtmd_params = MtmdContextParams {
            use_gpu: use_gpu && params.n_gpu_layers() != 0,
            ..MtmdContextParams::default()
        };
        let mmproj = mmproj_path.to_string_lossy();
        let mtmd = MtmdContext::init_from_file(&mmproj, &model, &mtmd_params).map_err(local_error)?;
        let template = model.chat_template(None).map_err(local_error)?;
        Ok(Self { backend, model, mtmd, template, sampling: Sampling::default() })
    }

    /// Sets how replies are generated.
    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    /// Runs on the calling thread; llama.cpp is not async, so this blocks
    /// until the reply is complete. Call it from a thread of its own.
    pub(crate) async fn prompt_model(
        &self,
        user_prompt: &str,
        images: &[&DynamicImage],
        system_prompt: Option<&str>,
    ) -> Result<String> {
        // Images are marked in the text and passed in the same order.
        // Without placeholders, the images come first (as in Python).
        let marker = mtmd_default_marker();
        let segments = segments(user_prompt);
        let mut text = String::new();
        let mut ordered: Vec<&DynamicImage> = Vec::new();
        if !segments.iter().any(|s| matches!(s, Segment::Image(_))) {
            for image in images {
                text.push_str(marker);
                ordered.push(image);
            }
            text.push_str(user_prompt);
        } else {
            for segment in segments {
                match segment {
                    Segment::Text(t) => text.push_str(t),
                    Segment::Image(i) => {
                        if let Some(image) = images.get(i) {
                            text.push_str(marker);
                            ordered.push(image);
                        }
                    }
                }
            }
        }

        let mut messages = Vec::new();
        if let Some(system_prompt) = system_prompt.filter(|s| !s.is_empty()) {
            messages.push(("system", system_prompt));
        }
        messages.push(("user", &text));
        let prompt = self.apply_chat_template(&messages)?;
        log::debug!("prompt: {prompt}");

        let bitmaps = ordered
            .iter()
            .map(|image| {
                let rgb = image.to_rgb8();
                MtmdBitmap::from_image_data(rgb.width(), rgb.height(), rgb.as_raw())
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(local_error)?;
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let chunks = self
            .mtmd
            .tokenize(
                MtmdInputText { text: prompt, add_special: true, parse_special: true },
                &bitmap_refs,
            )
            .map_err(local_error)?;

        let mut context = self
            .model
            .new_context(&self.backend, context_params())
            .map_err(local_error)?;
        let mut n_past = chunks
            .eval_chunks(&self.mtmd, &context, 0, 0, BATCH_SIZE as i32, true)
            .map_err(local_error)?;

        let mut sampler = sampler(&self.sampling);
        let vocab = self.model.vocab();
        let mut output = Vec::new();
        let mut batch = LlamaBatch::new(1, 1);
        for _ in 0..self.sampling.max_new_tokens {
            let token = sampler.sample(&context, -1);
            sampler.accept(token);
            if vocab.is_eog(token) {
                break;
            }
            output.extend(vocab.token_to_piece(token, false, None));
            batch.clear();
            batch.add(token, n_past, &[0], true).map_err(local_error)?;
            context.decode(&mut batch).map_err(local_error)?;
            n_past += 1;
        }

        let text = String::from_utf8_lossy(&output).trim().to_string();
        if text.is_empty() {
            return Err(Error::EmptyResponse { provider: "llama.cpp" });
        }
        Ok(text)
    }

    /// Formats `(role, content)` messages as a prompt. llama.cpp's own
    /// template engine only knows a fixed list of chat formats (not, e.g.,
    /// Gemma 4's), so other templates are rendered as Jinja, like Hugging
    /// Face's `apply_chat_template` does.
    fn apply_chat_template(&self, messages: &[(&str, &str)]) -> Result<String> {
        let chat = messages
            .iter()
            .map(|&(role, content)| LlamaChatMessage::new(role.into(), content.into()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(local_error)?;
        if let Ok(prompt) = self.model.apply_chat_template(&self.template, &chat, true) {
            return Ok(prompt);
        }
        let template = self.template.to_str().map_err(local_error)?;
        render_jinja_template(template, messages).map_err(local_error)
    }
}

fn context_params() -> LlamaContextParams {
    LlamaContextParams::default().with_n_ctx(NonZeroU32::new(CONTEXT_SIZE)).with_n_batch(BATCH_SIZE)
}

/// Lets llama.cpp choose how many layers go to the GPU so that the model, a
/// context and the image encoder fit into its free memory. Returns false if
/// there is no way to (including when there is no GPU), and the model should
/// run on the CPU.
fn fit_to_memory(params: Pin<&mut LlamaModelParams>, model_path: &Path, mmproj_path: &Path) -> bool {
    let Ok(path) = CString::new(model_path.to_string_lossy().as_bytes()) else {
        return false;
    };
    let mmproj_size = std::fs::metadata(mmproj_path).map_or(0, |m| m.len() as usize);
    let mut margins = vec![mmproj_size + IMAGE_ENCODER_RESERVE; llama_cpp_2::max_devices()];
    match params.fit_params(&path, &mut context_params(), &mut margins, CONTEXT_SIZE, FIT_LOG_LEVEL) {
        Ok(_) => true,
        Err(e) => {
            log::warn!("model does not fit the GPU ({e}); using the CPU");
            false
        }
    }
}

/// Filters in the order Hugging Face's `generate` applies them: temperature,
/// then top-k, then top-p, then a random pick.
fn sampler(sampling: &Sampling) -> LlamaSampler {
    let mut samplers = vec![LlamaSampler::temp(sampling.temperature)];
    if let Some(k) = sampling.top_k {
        samplers.push(LlamaSampler::top_k(k));
    }
    if let Some(p) = sampling.top_p.filter(|&p| p < 1.0) {
        samplers.push(LlamaSampler::top_p(p, 1));
    }
    samplers.push(LlamaSampler::dist(rand_seed()));
    LlamaSampler::chain_simple(samplers)
}

fn render_jinja_template(
    template: &str,
    messages: &[(&str, &str)],
) -> std::result::Result<String, minijinja::Error> {
    let mut env = minijinja::Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // Python methods such as `dict.get` and `str.split`, used by templates.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |message: String| -> std::result::Result<String, _> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message))
    });
    env.add_template("chat", template)?;
    let messages: Vec<_> = messages
        .iter()
        .map(|&(role, content)| minijinja::context! { role, content })
        .collect();
    // No BOS token here: tokenizing adds it.
    env.get_template("chat")?.render(minijinja::context! {
        messages,
        add_generation_prompt => true,
        bos_token => "",
    })
}

fn rand_seed() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos())
}
