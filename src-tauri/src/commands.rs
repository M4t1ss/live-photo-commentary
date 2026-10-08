//! Tauri commands: the command-mapping table in RUSTIFICATION.md. Replies
//! that were separate WebSocket messages in the Python backend are return
//! values here; only `connect`'s initial burst and the pipeline's own events
//! go over the channel.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;
use tauri::State;
use tauri::ipc::Channel;

use crate::config::Config;
use crate::events::Event;
use crate::promptsets::{DEFAULT_NAME, Promptset};
use crate::state::AppState;

/// `{name, fields, names}`, the shape `load_promptset`/`delete_promptset`
/// return and send as `promptset_loaded`.
#[derive(Serialize)]
pub struct PromptsetPayload {
    pub name: String,
    pub fields: Promptset,
    pub names: Vec<String>,
}

/// `{name, names}`, what `upload_model` returns.
#[derive(Serialize)]
pub struct UploadPayload {
    pub name: String,
    pub names: Vec<String>,
}

/// Stores the event channel, replacing any previous one (a page reload),
/// and sends the initial burst Python sends on WS connect: `config`,
/// `ready_state`, one `error` per stored model error, `models` (once voice
/// listing finishes) and `promptset_loaded`. Reconnecting also stops the
/// cycle, standing in for the WS disconnect Python reacts to.
#[tauri::command]
pub fn connect(state: State<'_, Arc<AppState>>, events: Channel<serde_json::Value>) -> Result<(), String> {
    state.pipeline().stop_cycle();
    state.set_channel(events);

    let cfg = state.config();
    state.send(Event::Config { data: cfg.masked() });
    state.send_ready_state();
    for message in state.model_errors() {
        state.send(Event::Error { message });
    }

    let app_state = Arc::clone(state.inner());
    tauri::async_runtime::spawn(async move { app_state.send_models().await });

    let name = cfg.active_promptset;
    let fields = state.promptsets().load(&name)?;
    let names = state.promptsets().list_names()?;
    state.send(Event::PromptsetLoaded { name, fields, names });
    Ok(())
}

#[tauri::command]
pub fn get_config(state: State<'_, Arc<AppState>>) -> Config {
    state.config().masked()
}

/// Reloads the VLM and/or TTS exactly when Python's websocket handler does
/// (`state::reload_triggers`).
#[tauri::command]
pub fn set_config(
    state: State<'_, Arc<AppState>>,
    data: serde_json::Map<String, serde_json::Value>,
    persist: bool,
) -> Result<Config, String> {
    let (old, new) = state.apply_config(&data)?;
    if persist {
        new.save(state.settings_path())?;
    }

    let triggers = crate::state::reload_triggers(&old, &new);
    let app_state = Arc::clone(state.inner());
    if triggers.vlm {
        let app_state = Arc::clone(&app_state);
        tauri::async_runtime::spawn(async move { app_state.reinit_describer().await });
    }
    if triggers.tts {
        tauri::async_runtime::spawn(async move { app_state.reinit_synthesizer().await });
    }

    state.send(Event::Config { data: new.masked() });
    Ok(new.masked())
}

/// Lists Kokoro's voices and sends `models`.
#[tauri::command]
pub async fn get_models(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.send_models().await;
    Ok(())
}

#[tauri::command]
pub fn start_cycle(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.pipeline().start_cycle();
    Ok(())
}

#[tauri::command]
pub fn stop_cycle(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.pipeline().stop_cycle();
    Ok(())
}

#[tauri::command]
pub fn list_promptsets(state: State<'_, Arc<AppState>>) -> Result<Vec<String>, String> {
    state.promptsets().list_names()
}

/// Settings "OK": applies the given prompt fields to the live describer for
/// this session only, without saving them as a named promptset, then
/// re-triggers pre-generation.
#[tauri::command]
pub fn apply_prompts(state: State<'_, Arc<AppState>>, fields: HashMap<String, String>) -> Result<(), String> {
    Arc::clone(state.inner()).apply_prompt_override(&fields);
    Ok(())
}

#[tauri::command]
pub fn load_promptset(state: State<'_, Arc<AppState>>, name: Option<String>) -> Result<PromptsetPayload, String> {
    let name = name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()).unwrap_or_else(|| DEFAULT_NAME.to_string());
    let fields = Arc::clone(state.inner()).load_promptset(&name)?;
    let names = state.promptsets().list_names()?;
    Ok(PromptsetPayload { name, fields, names })
}

#[tauri::command]
pub fn save_promptset(state: State<'_, Arc<AppState>>, name: String, fields: Promptset) -> Result<Vec<String>, String> {
    let name = name.trim().to_string();
    if name.is_empty() || name == DEFAULT_NAME {
        return Err("Cannot save as 'default'".to_string());
    }
    state.promptsets().save(&name, &fields)?;
    if state.config().active_promptset == name {
        state.apply_active_promptset_to_describer();
    }
    state.promptsets().list_names()
}

#[tauri::command]
pub fn delete_promptset(state: State<'_, Arc<AppState>>, name: String) -> Result<PromptsetPayload, String> {
    let name = name.trim().to_string();
    if name.is_empty() || name == DEFAULT_NAME {
        return Err("Cannot delete 'default'".to_string());
    }
    let (active, fields) = Arc::clone(state.inner()).delete_promptset(&name)?;
    let names = state.promptsets().list_names()?;
    Ok(PromptsetPayload { name: active, fields, names })
}

/// Always ends with `system_tts_done` on the channel, also when there's
/// nothing to say (handled inside `Pipeline::synthesize_system`).
#[tauri::command]
pub fn synthesize(state: State<'_, Arc<AppState>>, text: String) -> Result<(), String> {
    state.pipeline().synthesize_system(text);
    Ok(())
}

/// Same `system_tts_done` guarantee as `synthesize`, for a pre-generated
/// message instead of ad-hoc text.
#[tauri::command]
pub fn play_system_message(state: State<'_, Arc<AppState>>, name: String) -> Result<(), String> {
    state.pipeline().play_system_message(&name);
    Ok(())
}

/// The active avatar model's camelCase configuration, plus the absolute
/// `modelPath`/`animationsDir` the frontend will pass to `convertFileSrc`
/// (phase 5).
#[tauri::command]
pub fn model_config(state: State<'_, Arc<AppState>>) -> Result<serde_json::Value, String> {
    let active = state.config().active_model;
    let mut config = state.avatar_models().model_config(&active)?;
    let (model_path, _) = state.avatar_models().resolve(&active);
    if let serde_json::Value::Object(map) = &mut config {
        map.insert("modelPath".to_string(), serde_json::json!(model_path));
        map.insert("animationsDir".to_string(), serde_json::json!(state.animations_dir()));
    }
    Ok(config)
}

#[tauri::command]
pub fn avatar_models(state: State<'_, Arc<AppState>>) -> Vec<String> {
    state.avatar_models().list_names()
}

#[tauri::command]
pub fn open_model_dir(app: tauri::AppHandle, state: State<'_, Arc<AppState>>) -> Result<(), String> {
    let dir = state.avatar_models().user_dir().to_path_buf();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    use tauri_plugin_opener::OpenerExt;
    app.opener().open_path(dir.to_string_lossy(), None::<&str>).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn upload_model(request: tauri::ipc::Request<'_>, state: State<'_, Arc<AppState>>) -> Result<UploadPayload, String> {
    let filename = request
        .headers()
        .get("X-Filename")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("model.glb")
        .to_string();
    let body = match request.body() {
        tauri::ipc::InvokeBody::Raw(bytes) => bytes.as_slice(),
        _ => return Err("expected a raw request body".to_string()),
    };
    let name = state.avatar_models().save_user_model(&filename, body)?;
    let names = state.avatar_models().list_names();
    Ok(UploadPayload { name, names })
}

/// Which GPU backend local models run on, and the state of the optional CUDA
/// backend, for the Settings "GPU acceleration" section.
#[tauri::command]
pub fn gpu_status(app: tauri::AppHandle) -> Result<crate::gpu::GpuStatus, String> {
    crate::gpu::status(&app)
}

/// Downloads and installs the CUDA backend (progress as `download_progress`
/// events), then reloads a local model so it runs on it.
#[tauri::command]
pub async fn download_cuda_backend(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<crate::gpu::GpuStatus, String> {
    let (Some(url), Some(sha256)) = (crate::cuda_backend::download_url(), crate::cuda_backend::SHA256) else {
        return Err("this build of the app cannot download the CUDA backend".to_string());
    };
    let root = crate::gpu::cuda_root(&app)?;
    let reporter = Arc::clone(state.inner());
    crate::cuda_backend::install(&root, &url, sha256, move |downloaded, total| {
        reporter.send(Event::DownloadProgress { file: "CUDA backend".to_string(), downloaded, total });
    })
    .await?;
    if state.config().vlm_provider.as_deref() == Some("local") {
        state.inner().reinit_describer().await;
    }
    crate::gpu::status(&app)
}

/// Deletes the downloaded CUDA backend; if it is in use, at the next start.
#[tauri::command]
pub fn remove_cuda_backend(app: tauri::AppHandle) -> Result<crate::gpu::GpuStatus, String> {
    crate::cuda_backend::remove(&crate::gpu::cuda_root(&app)?)?;
    crate::gpu::status(&app)
}
