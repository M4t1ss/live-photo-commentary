//! Session state and model lifecycle: which reload a `set_config` call should
//! trigger, the ad-hoc prompt overrides from the Settings "OK" button, and the
//! pre-generation sequence counter.
//!
//! Describer construction (`_make_describer`): the cloud providers are pure;
//! `local` downloads the catalogue entry's GGUF files (reporting progress)
//! and loads them with llama.cpp.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use kokoro_timestamped::{KokoroSynthesizer, Voice, VoiceBlendPart};
use tauri::Manager;

use crate::avatar::AvatarModels;
use crate::catalogue::{Catalogue, CatalogueEntry};
use crate::config::Config;
use crate::events::{Event, ModelKind};
use crate::pipeline::Pipeline;
use crate::promptsets::{Promptset, Promptsets};

/// The five fields pushed onto the live describer.
const DESCRIBER_FIELDS: [&str; 5] = ["system_prompt", "prompt", "first_prompt", "history_prompt", "compact_prompt"];
/// The three system-message prompts.
const SYSTEM_MESSAGE_FIELDS: [&str; 3] = ["greeting_prompt", "farewell_prompt", "lonely_prompt"];

/// Which models a `set_config` call should reload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReloadTriggers {
    pub vlm: bool,
    pub tts: bool,
}

/// Decides which models need reloading after `old` becomes `new`: the VLM
/// provider, model or overrides changed, or the active provider's own API
/// key did; the TTS voice changed.
pub fn reload_triggers(old: &Config, new: &Config) -> ReloadTriggers {
    let api_key_changed = match new.vlm_provider.as_deref() {
        Some("gemini") => old.gemini_api_key != new.gemini_api_key,
        Some("openai") => {
            old.openai_api_key != new.openai_api_key
                || old.openai_compat_api_key != new.openai_compat_api_key
                || old.openai_base_url != new.openai_base_url
        }
        _ => false,
    };
    let vlm = old.vlm_provider != new.vlm_provider
        || old.vlm_model != new.vlm_model
        || old.vlm_model_overrides != new.vlm_model_overrides
        || api_key_changed;
    let tts = old.tts_voice != new.tts_voice;
    ReloadTriggers { vlm, tts }
}

/// Parses `tts_voice` as a plain voice name, or as a JSON object of
/// `{voice: weight}` for a blend.
pub fn parse_voice(raw: &str) -> Voice {
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(raw) {
        let parts = map
            .into_iter()
            .filter_map(|(name, weight)| weight.as_f64().map(|weight| VoiceBlendPart { name, weight: weight as f32 }))
            .collect();
        return Voice::Blend(parts);
    }
    Voice::Name(raw.to_string())
}

/// Loads the Kokoro synthesizer for `cfg.tts_voice` (always model `"model"`,
/// speed 1.0, language from the voice name).
pub async fn build_synthesizer(cfg: &Config) -> Result<kokoro_timestamped::KokoroSynthesizer, String> {
    kokoro_timestamped::KokoroSynthesizer::new(parse_voice(&cfg.tts_voice), "model", None, 1.0)
        .await
        .map_err(|e| e.to_string())
}

/// Builds the backend for a cloud VLM provider.
fn build_cloud_backend(cfg: &Config) -> Result<vlm_describer::Backend, String> {
    let model = cfg.vlm_model.clone().ok_or_else(|| "no VLM model configured".to_string())?;
    match cfg.vlm_provider.as_deref() {
        Some("gemini") => {
            let api_key = cfg.gemini_api_key.clone().unwrap_or_default();
            Ok(vlm_describer::Backend::Gemini(vlm_describer::Gemini::new(api_key, model)))
        }
        Some("openai") => {
            let api_key = if cfg.openai_base_url.is_some() {
                cfg.openai_compat_api_key.clone()
            } else {
                cfg.openai_api_key.clone()
            };
            let mut openai = vlm_describer::OpenAi::new(api_key, model);
            if let Some(base_url) = &cfg.openai_base_url {
                openai = openai.with_base_url(base_url.clone());
            }
            Ok(vlm_describer::Backend::OpenAi(openai))
        }
        Some(other) => Err(format!("unknown VLM provider {other:?}")),
        None => Err("no VLM provider configured".to_string()),
    }
}

/// Downloads (if needed) and loads a local model. Downloading is async;
/// loading takes seconds of CPU and disk, so it runs on a blocking thread.
/// `gpu_layers` is how many layers go to the GPU.
async fn build_local_backend(
    entry: &CatalogueEntry,
    progress: vlm_describer::DownloadProgress,
    gpu_layers: u32,
) -> Result<vlm_describer::Backend, String> {
    let files = entry.gguf.as_ref().ok_or_else(|| {
        format!(
            "'{}' is not in the catalogue; add gguf_repo, model_file and mmproj_file to the model overrides",
            entry.name
        )
    })?;
    let (model, mmproj) = vlm_describer::download_model(&files.repo, &files.model_file, &files.mmproj_file, Some(progress))
        .await
        .map_err(|e| e.to_string())?;
    let sampling = entry.sampling();
    let loaded = tokio::task::spawn_blocking(move || {
        vlm_describer::LlamaCpp::new(&model, &mmproj, gpu_layers).map(|llama| llama.with_sampling(sampling))
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    Ok(vlm_describer::Backend::LlamaCpp(loaded))
}

/// Builds the describer for `cfg`: the backend for its provider, with the
/// catalogue entry's `response_re` (and the "Model overrides" setting merged
/// over the entry). An old local model must be dropped before this is
/// called, because llama.cpp's backend is global.
pub async fn build_describer(
    cfg: &Config,
    catalogue: &Catalogue,
    progress: vlm_describer::DownloadProgress,
    gpu_layers: u32,
) -> Result<vlm_describer::Describer, String> {
    let model = cfg.vlm_model.clone().ok_or_else(|| "no VLM model configured".to_string())?;
    let mut entry = catalogue.entry_for_model(&model);
    if let Some(overrides) = &cfg.vlm_model_overrides {
        entry = entry.with_overrides(overrides);
    }
    let backend = match cfg.vlm_provider.as_deref() {
        Some("local") => build_local_backend(&entry, progress, gpu_layers).await?,
        _ => build_cloud_backend(cfg)?,
    };
    let describer = vlm_describer::Describer::new(backend);
    match &entry.response_re {
        Some(pattern) => describer.with_response_re(pattern).map_err(|e| format!("invalid response_re: {e}")),
        None => Ok(describer),
    }
}

/// Ad-hoc prompt overrides from the Settings "OK" button: applied to the live
/// describer and to pre-generation for this session only, cleared by an
/// explicit promptset load or delete.
#[derive(Default)]
pub struct SessionState {
    prompt_override: Option<HashMap<String, String>>,
    sys_msg_override: HashMap<String, String>,
}

impl SessionState {
    /// Splits `given` into the describer-field override and the
    /// system-message-field override, dropping anything else.
    pub fn set_override(&mut self, given: &HashMap<String, String>) {
        let describer_fields: HashMap<String, String> = given
            .iter()
            .filter(|(key, _)| DESCRIBER_FIELDS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        self.prompt_override = if describer_fields.is_empty() { None } else { Some(describer_fields) };
        self.sys_msg_override = given
            .iter()
            .filter(|(key, _)| SYSTEM_MESSAGE_FIELDS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
    }

    /// Clears both overrides (loading or deleting the active promptset).
    pub fn clear(&mut self) {
        self.prompt_override = None;
        self.sys_msg_override.clear();
    }

    /// The five describer fields: `base` (the active promptset) with the
    /// ad-hoc override layered on top, tag-substituted.
    pub fn describer_fields(&self, base: &Promptset, tag_names: &[&str]) -> Promptset {
        let mut fields = base.clone();
        if let Some(over) = &self.prompt_override {
            apply_fields(&mut fields, over);
        }
        fields.substitute_tags(tag_names)
    }

    /// All eight fields, merged for pre-generation and tag-substituted. Every
    /// field is substituted before the describer fields and the three message
    /// prompts are read from the result, so a custom greeting, farewell or
    /// lonely prompt that itself contains the tag placeholder gets it
    /// substituted too (the defaults never contain it).
    pub fn pregen_fields(&self, base: &Promptset, tag_names: &[&str]) -> Promptset {
        let mut fields = base.clone();
        if let Some(over) = &self.prompt_override {
            apply_fields(&mut fields, over);
        }
        apply_fields(&mut fields, &self.sys_msg_override);
        fields.substitute_tags(tag_names)
    }
}

fn apply_fields(target: &mut Promptset, values: &HashMap<String, String>) {
    for (key, value) in values {
        match key.as_str() {
            "system_prompt" => target.system_prompt = value.clone(),
            "prompt" => target.prompt = value.clone(),
            "first_prompt" => target.first_prompt = value.clone(),
            "history_prompt" => target.history_prompt = value.clone(),
            "compact_prompt" => target.compact_prompt = value.clone(),
            "greeting_prompt" => target.greeting_prompt = value.clone(),
            "farewell_prompt" => target.farewell_prompt = value.clone(),
            "lonely_prompt" => target.lonely_prompt = value.clone(),
            _ => {}
        }
    }
}

/// The three system-message prompts, keyed by name without the `_prompt`
/// suffix, for
/// [`crate::pipeline::Pipeline::pregen_system_messages`].
pub fn system_message_prompts(fields: &Promptset) -> HashMap<String, String> {
    HashMap::from([
        ("greeting".to_string(), fields.greeting_prompt.clone()),
        ("farewell".to_string(), fields.farewell_prompt.clone()),
        ("lonely".to_string(), fields.lonely_prompt.clone()),
    ])
}

/// True when none of the three system-message prompts has any text: pregen
/// should be skipped entirely and `system_messages` marked ready right away.
pub fn system_messages_are_empty(fields: &Promptset) -> bool {
    [&fields.greeting_prompt, &fields.farewell_prompt, &fields.lonely_prompt].iter().all(|p| p.trim().is_empty())
}

/// Ignores a stale pre-generation completion racing a newer run.
#[derive(Default)]
pub struct PregenSequencer {
    seq: AtomicU64,
}

impl PregenSequencer {
    /// Starts a new run, returning its sequence number.
    pub fn begin(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Whether `seq` (from an earlier `begin`) is still the latest run —
    /// i.e. whether its completion should still be acted on.
    pub fn is_current(&self, seq: u64) -> bool {
        self.seq.load(Ordering::SeqCst) == seq
    }
}

/// The real pipeline type the running app uses (as opposed to the fakes
/// `pipeline.rs`'s own tests drive it with).
pub type AppPipeline = Pipeline<vlm_describer::Describer, KokoroSynthesizer>;

/// The folder the Python backend of versions before 0.2.0 was installed in.
pub fn old_python_backend_dir(app_data_dir: &std::path::Path) -> PathBuf {
    app_data_dir.join("backend")
}

/// Where bundled avatar models live: the repo's own folder in debug, the
/// bundled resources in release (`tauri.bundle.conf.json`'s `resources` list;
/// mirrors `locate_screenshot_exe` in `screenshot.rs`).
fn bundled_models_dir(app: &tauri::AppHandle) -> PathBuf {
    if cfg!(debug_assertions) {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().expect("CARGO_MANIFEST_DIR should have a parent").join("models")
    } else {
        // Tauri's resource folder starts with `\\?\` on Windows, which makes Windows
        // read `/` literally, and the frontend joins file names to the folder with
        // `/` (`animationsDir` in `avatar.js`): none of the animations would load.
        let resources = app.path().resource_dir().expect("resource dir should be available");
        dunce::simplified(&resources).join("resources").join("models")
    }
}

/// `.vrma` animation clips referenced by avatar YAML files, in `animations/`
/// under the models (bundled by `tauri.bundle.conf.json`'s
/// `resources/models/animations/*`).
fn animations_dir(app: &tauri::AppHandle) -> PathBuf {
    bundled_models_dir(app).join("animations")
}

/// Everything the Tauri commands (`commands.rs`) need: settings, promptsets,
/// avatar models, the VLM catalogue, the pipeline, the event channel, and
/// the model-lifecycle/session state. Managed as `Arc<AppState>` so
/// background work (model loads,
/// pre-generation) can hold its own reference.
pub struct AppState {
    config: Mutex<Config>,
    settings_path: PathBuf,
    promptsets: Promptsets,
    avatar_models: AvatarModels,
    catalogue: Catalogue,
    pipeline: Arc<AppPipeline>,
    /// Shared with the `EventSink` closure `pipeline` was built with, so
    /// `set_channel` can change what that closure sends to.
    channel: Arc<Mutex<Option<tauri::ipc::Channel<serde_json::Value>>>>,
    session: Mutex<SessionState>,
    pregen_sequencer: PregenSequencer,
    vlm_ready: AtomicBool,
    tts_ready: AtomicBool,
    system_messages_ready: AtomicBool,
    model_errors: Mutex<HashMap<String, String>>,
    model_dir: PathBuf,
    animations_dir: PathBuf,
}

impl AppState {
    /// Loads settings (migrating `.env` once if needed), promptsets and user
    /// avatar models (migrating from the old backend's folders once), and
    /// the VLM catalogue; starts the VLM/TTS worker threads with no model
    /// loaded yet. Pure local filesystem work — no network, nothing async.
    pub fn new(app: &tauri::AppHandle) -> Result<AppState, String> {
        let settings_path = app.path().app_config_dir().map_err(|e| e.to_string())?.join("settings.json");
        let app_data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
        // The Python-era install kept its settings, promptsets and uploaded
        // avatar models in `<app data>/backend`.
        let old_backend_dir = old_python_backend_dir(&app_data_dir);
        let config = Config::load_or_migrate(&settings_path, &old_backend_dir.join(".env"))?;

        let prompts_dir = app_data_dir.join("prompts");
        crate::promptsets::migrate(&old_backend_dir.join("prompts"), &prompts_dir)?;
        let promptsets = Promptsets::new(prompts_dir)?;

        let user_models_dir = app_data_dir.join("user_models");
        crate::avatar::migrate_user_models(&old_backend_dir.join("user_models"), &user_models_dir)?;
        let model_dir = bundled_models_dir(app);
        let avatar_models = AvatarModels::new(model_dir.clone(), user_models_dir)?;

        let catalogue = Catalogue::parse(include_str!("../vlm_models.yaml"))?;

        let channel: Arc<Mutex<Option<tauri::ipc::Channel<serde_json::Value>>>> = Arc::new(Mutex::new(None));
        let sink: crate::events::EventSink = {
            let channel = Arc::clone(&channel);
            Arc::new(move |event: Event| {
                if let Some(channel) = channel.lock().expect("channel mutex should not be poisoned").as_ref() {
                    let _ = channel.send(event.to_json());
                }
            })
        };
        let pipeline = Arc::new(AppPipeline::new(sink));

        Ok(AppState {
            config: Mutex::new(config),
            settings_path,
            promptsets,
            avatar_models,
            catalogue,
            pipeline,
            channel,
            session: Mutex::new(SessionState::default()),
            pregen_sequencer: PregenSequencer::default(),
            vlm_ready: AtomicBool::new(false),
            tts_ready: AtomicBool::new(false),
            system_messages_ready: AtomicBool::new(true),
            model_errors: Mutex::new(HashMap::new()),
            model_dir,
            animations_dir: animations_dir(app),
        })
    }

    pub fn pipeline(&self) -> &Arc<AppPipeline> {
        &self.pipeline
    }

    pub fn promptsets(&self) -> &Promptsets {
        &self.promptsets
    }

    pub fn avatar_models(&self) -> &AvatarModels {
        &self.avatar_models
    }

    pub fn model_dir(&self) -> &PathBuf {
        &self.model_dir
    }

    pub fn animations_dir(&self) -> &PathBuf {
        &self.animations_dir
    }

    pub fn config(&self) -> Config {
        self.config.lock().expect("config mutex should not be poisoned").clone()
    }

    pub fn settings_path(&self) -> &PathBuf {
        &self.settings_path
    }

    /// Replaces `set_config`'s updates onto the stored config, returning
    /// `(old, new)`. Doesn't persist or reload anything — the caller (the
    /// `set_config` command) decides that.
    pub fn apply_config(&self, updates: &serde_json::Map<String, serde_json::Value>) -> Result<(Config, Config), String> {
        let mut guard = self.config.lock().expect("config mutex should not be poisoned");
        let old = guard.clone();
        let new = old.apply(updates)?;
        *guard = new.clone();
        Ok((old, new))
    }

    pub fn set_channel(&self, channel: tauri::ipc::Channel<serde_json::Value>) {
        *self.channel.lock().expect("channel mutex should not be poisoned") = Some(channel);
    }

    pub fn send(&self, event: Event) {
        if let Some(channel) = self.channel.lock().expect("channel mutex should not be poisoned").as_ref() {
            let _ = channel.send(event.to_json());
        }
    }

    pub fn send_ready_state(&self) {
        let vlm = self.vlm_ready.load(Ordering::SeqCst);
        let tts = self.tts_ready.load(Ordering::SeqCst);
        let system_messages = self.system_messages_ready.load(Ordering::SeqCst);
        self.send(Event::ReadyState {
            ready: vlm && tts && system_messages,
            vlm,
            tts,
            system_messages,
            running: self.pipeline.running(),
        });
    }

    pub fn model_errors(&self) -> Vec<String> {
        self.model_errors.lock().expect("model_errors mutex should not be poisoned").values().cloned().collect()
    }

    /// Lists Kokoro's voices and sends `models`.
    pub async fn send_models(&self) {
        let voices = match KokoroSynthesizer::list_voices().await {
            Ok(voices) => voices,
            Err(e) => {
                log::warn!("Could not list Kokoro voices: {e}");
                Vec::new()
            }
        };
        let vlm = self.catalogue.providers().iter().map(|(provider, entries)| (provider.clone(), entries.clone())).collect();
        self.send(Event::Models { vlm, tts: voices });
    }

    fn active_promptset(&self) -> Promptset {
        let name = self.config().active_promptset;
        self.promptsets.load(&name).unwrap_or_default()
    }

    fn active_tag_names(&self) -> Vec<String> {
        let model = self.config().active_model;
        self.avatar_models.tag_names(&model).unwrap_or_default()
    }

    /// The five describer fields for the active promptset, with the
    /// session's ad-hoc override layered on top.
    pub fn describer_fields(&self) -> Promptset {
        let base = self.active_promptset();
        let tag_names = self.active_tag_names();
        let tag_refs: Vec<&str> = tag_names.iter().map(String::as_str).collect();
        self.session.lock().expect("session mutex should not be poisoned").describer_fields(&base, &tag_refs)
    }

    fn pregen_fields(&self) -> Promptset {
        let base = self.active_promptset();
        let tag_names = self.active_tag_names();
        let tag_refs: Vec<&str> = tag_names.iter().map(String::as_str).collect();
        self.session.lock().expect("session mutex should not be poisoned").pregen_fields(&base, &tag_refs)
    }

    /// Pushes the active promptset (plus any session override) onto the
    /// live describer, without a model reload.
    pub fn apply_active_promptset_to_describer(&self) {
        self.pipeline.set_prompts(self.describer_fields().describer_prompts());
    }

    /// Sets the session's ad-hoc prompt override (Settings "OK"), applies it
    /// to the live describer, and re-triggers pre-generation.
    pub fn apply_prompt_override(self: &Arc<Self>, given: &HashMap<String, String>) {
        self.session.lock().expect("session mutex should not be poisoned").set_override(given);
        self.apply_active_promptset_to_describer();
        self.trigger_pregen();
    }

    /// Clears the session override, persists `name` as the active promptset,
    /// pushes it onto the describer, resets the describer's history, and
    /// re-triggers pre-generation. Returns the loaded fields for the
    /// `promptset_loaded` event.
    pub fn load_promptset(self: &Arc<Self>, name: &str) -> Result<Promptset, String> {
        let fields = self.promptsets.load(name)?;
        self.session.lock().expect("session mutex should not be poisoned").clear();
        {
            let mut cfg = self.config.lock().expect("config mutex should not be poisoned");
            *cfg = cfg.apply(&serde_json::Map::from_iter([("active_promptset".to_string(), serde_json::json!(name))]))?;
        }
        self.config().save(&self.settings_path)?;
        self.apply_active_promptset_to_describer();
        self.pipeline.reset_describer_history();
        self.trigger_pregen();
        Ok(fields)
    }

    /// Deletes `name`; if it was active, falls back to `default` the same
    /// way `load_promptset` does. Returns the now-active name and its fields.
    pub fn delete_promptset(self: &Arc<Self>, name: &str) -> Result<(String, Promptset), String> {
        let was_active = self.config().active_promptset == name;
        self.promptsets.delete(name)?;
        if was_active {
            self.load_promptset(crate::promptsets::DEFAULT_NAME)?;
        }
        let active = self.config().active_promptset;
        let fields = self.promptsets.load(&active)?;
        Ok((active, fields))
    }

    /// Generates and synthesizes the greeting/farewell/lonely messages if
    /// both models are ready and at least one of them has text.
    pub fn trigger_pregen(self: &Arc<Self>) {
        if !(self.vlm_ready.load(Ordering::SeqCst) && self.tts_ready.load(Ordering::SeqCst)) {
            return;
        }
        let fields = self.pregen_fields();
        if system_messages_are_empty(&fields) {
            self.system_messages_ready.store(true, Ordering::SeqCst);
            self.send_ready_state();
            return;
        }
        self.system_messages_ready.store(false, Ordering::SeqCst);
        let seq = self.pregen_sequencer.begin();
        self.send(Event::LoadProgress { message: "Generating messages…".to_string() });
        self.send_ready_state();

        let prompts = system_message_prompts(&fields);
        let system_prompt = fields.system_prompt.clone();
        let this = Arc::clone(self);
        Arc::clone(&self.pipeline).pregen_system_messages(prompts, system_prompt, move || {
            if !this.pregen_sequencer.is_current(seq) {
                return;
            }
            this.system_messages_ready.store(true, Ordering::SeqCst);
            this.send_ready_state();
        });
    }

    /// (Re)loads the VLM describer for the current config, reporting
    /// progress and errors on the channel.
    /// A no-op (besides a `ready_state`) when no provider or model is
    /// configured yet.
    pub async fn reinit_describer(self: &Arc<Self>) {
        self.model_errors.lock().expect("model_errors mutex should not be poisoned").remove("vlm");
        let had_describer = self.vlm_ready.swap(false, Ordering::SeqCst);
        let cfg = self.config();
        if cfg.vlm_provider.is_none() || cfg.vlm_model.is_none() {
            self.send_ready_state();
            return;
        }
        self.send(Event::ModelLoading { model: ModelKind::Vlm });
        self.send_ready_state();

        if had_describer {
            self.send(Event::LoadProgress { message: "Waiting for current generation to finish…".to_string() });
        }
        self.set_describer_blocking(None).await;
        if had_describer {
            self.send(Event::LoadProgress { message: "Releasing previous model…".to_string() });
        }

        let progress: vlm_describer::DownloadProgress = {
            let this = Arc::clone(self);
            Arc::new(move |file, downloaded, total| {
                this.send(Event::DownloadProgress { file: file.to_string(), downloaded, total });
            })
        };
        match build_describer(&cfg, &self.catalogue, progress, crate::gpu::gpu_layers()).await {
            Ok(describer) => {
                self.set_describer_blocking(Some(describer)).await;
                self.apply_active_promptset_to_describer();
                self.vlm_ready.store(true, Ordering::SeqCst);
                self.send(Event::ModelReady { model: ModelKind::Vlm });
                self.send_ready_state();
                if self.tts_ready.load(Ordering::SeqCst) {
                    self.trigger_pregen();
                }
            }
            Err(message) => {
                let message = format!("VLM init failed: {message}");
                self.model_errors.lock().expect("model_errors mutex should not be poisoned").insert("vlm".to_string(), message.clone());
                self.send(Event::Error { message });
                self.send_ready_state();
            }
        }
    }

    /// (Re)loads the Kokoro synthesizer for the current config.
    pub async fn reinit_synthesizer(self: &Arc<Self>) {
        self.set_synthesizer_blocking(None).await;
        self.model_errors.lock().expect("model_errors mutex should not be poisoned").remove("tts");
        self.tts_ready.store(false, Ordering::SeqCst);
        self.send(Event::ModelLoading { model: ModelKind::Tts });
        self.send_ready_state();

        let cfg = self.config();
        match build_synthesizer(&cfg).await {
            Ok(synthesizer) => {
                self.set_synthesizer_blocking(Some(synthesizer)).await;
                self.tts_ready.store(true, Ordering::SeqCst);
                self.send(Event::ModelReady { model: ModelKind::Tts });
                self.send_ready_state();
                if self.vlm_ready.load(Ordering::SeqCst) {
                    self.trigger_pregen();
                }
            }
            Err(message) => {
                let message = format!("TTS init failed: {message}");
                self.model_errors.lock().expect("model_errors mutex should not be poisoned").insert("tts".to_string(), message.clone());
                self.send(Event::Error { message });
                self.send_ready_state();
            }
        }
    }

    /// Loads both models in parallel at startup.
    pub async fn init_models(self: &Arc<Self>) {
        tokio::join!(self.reinit_describer(), self.reinit_synthesizer());
    }

    /// `set_describer`/`set_synthesizer` block the calling thread until the
    /// VLM/TTS worker acknowledges, so run them on a blocking thread when
    /// called from async code.
    async fn set_describer_blocking(&self, describer: Option<vlm_describer::Describer>) {
        let pipeline = Arc::clone(&self.pipeline);
        let _ = tokio::task::spawn_blocking(move || pipeline.set_describer(describer)).await;
    }

    async fn set_synthesizer_blocking(&self, synthesizer: Option<KokoroSynthesizer>) {
        let pipeline = Arc::clone(&self.pipeline);
        let _ = tokio::task::spawn_blocking(move || pipeline.set_synthesizer(synthesizer)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reload_triggers_on_provider_model_or_overrides_change() {
        let old = Config::default();
        let new = Config { vlm_model: Some("some/model".to_string()), ..Config::default() };
        assert_eq!(reload_triggers(&old, &new), ReloadTriggers { vlm: true, tts: false });

        let old = new.clone();
        let mut new = old.clone();
        new.vlm_model_overrides = Some("{}".to_string());
        assert_eq!(reload_triggers(&old, &new), ReloadTriggers { vlm: true, tts: false });
    }

    #[test]
    fn reload_triggers_on_the_active_providers_api_key_only() {
        let old = Config { vlm_provider: Some("gemini".to_string()), ..Config::default() };
        let mut new = old.clone();
        new.gemini_api_key = Some("new-key".to_string());
        assert!(reload_triggers(&old, &new).vlm, "the active provider's key changing must reload");

        let old = Config { vlm_provider: Some("gemini".to_string()), ..Config::default() };
        let mut new = old.clone();
        new.openai_api_key = Some("unrelated".to_string());
        assert!(!reload_triggers(&old, &new).vlm, "an inactive provider's key must not reload");
    }

    #[test]
    fn reload_triggers_on_tts_voice_change_only() {
        let old = Config::default();
        let mut new = old.clone();
        new.tts_voice = "af_nicole".to_string();
        let triggers = reload_triggers(&old, &new);
        assert!(triggers.tts);
        assert!(!triggers.vlm);
    }

    #[test]
    fn reload_triggers_are_both_false_when_nothing_relevant_changed() {
        let old = Config::default();
        let mut new = old.clone();
        new.idle_to_dance_secs = 999; // unrelated field
        assert_eq!(reload_triggers(&old, &new), ReloadTriggers::default());
    }

    #[test]
    fn parse_voice_plain_name_and_blend() {
        assert!(matches!(parse_voice("af_heart"), Voice::Name(name) if name == "af_heart"));
        match parse_voice(r#"{"af_nicole": 0.8, "jf_alpha": 0.2}"#) {
            Voice::Blend(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(parts.iter().any(|p| p.name == "af_nicole" && (p.weight - 0.8).abs() < 1e-6));
            }
            Voice::Name(_) | Voice::Array(_) => panic!("expected a blend"),
        }
    }

    #[test]
    fn build_cloud_backend_for_gemini_and_openai() {
        let cfg = Config {
            vlm_provider: Some("gemini".to_string()),
            vlm_model: Some("gemini-3.5-flash".to_string()),
            gemini_api_key: Some("secret".to_string()),
            ..Config::default()
        };
        assert!(matches!(build_cloud_backend(&cfg), Ok(vlm_describer::Backend::Gemini(_))));

        let cfg = Config {
            vlm_provider: Some("openai".to_string()),
            vlm_model: Some("gpt-4o".to_string()),
            ..Config::default()
        };
        assert!(matches!(build_cloud_backend(&cfg), Ok(vlm_describer::Backend::OpenAi(_))));
    }

    #[test]
    fn build_cloud_backend_reports_missing_config_clearly() {
        assert!(build_cloud_backend(&Config::default()).is_err());
        let cfg = Config { vlm_provider: Some("nope".to_string()), vlm_model: Some("m".to_string()), ..Config::default() };
        assert!(build_cloud_backend(&cfg).err().unwrap().contains("nope"));
    }

    fn no_progress() -> vlm_describer::DownloadProgress {
        Arc::new(|_, _, _| {})
    }

    #[tokio::test]
    async fn build_describer_applies_the_entrys_response_re() {
        let catalogue = Catalogue::parse("gemini:\n  - gemini-a\n").unwrap();
        let cfg = Config {
            vlm_provider: Some("gemini".to_string()),
            vlm_model: Some("gemini-a".to_string()),
            vlm_model_overrides: Some(r#"{"response_re": "(unclosed"}"#.to_string()),
            ..Config::default()
        };
        let error = build_describer(&cfg, &catalogue, no_progress(), 0).await.err().expect("a bad response_re should fail the build");
        assert!(error.contains("response_re"), "{error}");

        let cfg = Config { vlm_model_overrides: Some(r#"{"response_re": "(.*)"}"#.to_string()), ..cfg };
        assert!(build_describer(&cfg, &catalogue, no_progress(), 0).await.is_ok());
    }

    #[tokio::test]
    async fn a_local_model_outside_the_catalogue_needs_files_in_the_overrides() {
        let catalogue = Catalogue::parse("gemini:\n  - gemini-a\n").unwrap();
        let cfg = Config {
            vlm_provider: Some("local".to_string()),
            vlm_model: Some("my/model".to_string()),
            ..Config::default()
        };
        let error = build_describer(&cfg, &catalogue, no_progress(), 0).await.err().expect("a model without files should fail the build");
        assert!(error.contains("my/model") && error.contains("gguf_repo"), "{error}");
    }

    /// Loads the local catalogue entry named by `LOCAL_MODEL` (default
    /// `google/gemma-4-E2B-it`) through `build_describer`, as the app does,
    /// and describes the two sample frames, printing the times. Downloads the
    /// model on first use. `GPU_LAYERS` (default: `gpu_layers()`) picks CPU or
    /// GPU. Run on request, in release mode for CPU timings:
    /// `LOCAL_MODEL=Qwen/Qwen3-VL-2B-Instruct cargo test -p app --lib --features vulkan -- --ignored --nocapture real_local_model`
    #[tokio::test]
    #[ignore]
    async fn real_local_model_describes_the_sample_frames() {
        let frames_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let model = std::env::var("LOCAL_MODEL").unwrap_or_else(|_| "google/gemma-4-E2B-it".to_string());
        // `CUDA_BACKEND_DIR`: a folder with the CUDA backend, as unpacked by the
        // app, to run on instead of the installer's Vulkan.
        if let Some(dir) = std::env::var_os("CUDA_BACKEND_DIR") {
            crate::cuda_backend::load(std::path::Path::new(&dir));
        }
        vlm_describer::load_backends(None);
        let devices = vlm_describer::backend_devices();
        println!("devices: {devices:?}
model runs on: {:?}", vlm_describer::preferred_gpu(&devices));
        let gpu_layers = std::env::var("GPU_LAYERS").map_or_else(|_| crate::gpu::gpu_layers(), |v| v.parse().expect("GPU_LAYERS"));
        let catalogue = Catalogue::parse(include_str!("../vlm_models.yaml")).unwrap();
        let cfg = Config { vlm_provider: Some("local".to_string()), vlm_model: Some(model.clone()), ..Config::default() };
        let progress: vlm_describer::DownloadProgress =
            Arc::new(|file, done, total| eprintln!("downloading {file}: {done}/{total}"));

        let start = std::time::Instant::now();
        let mut describer = build_describer(&cfg, &catalogue, progress, gpu_layers).await.expect("load the model");
        println!("{model}: loaded in {:.1}s with {gpu_layers} GPU layers", start.elapsed().as_secs_f64());
        describer.prompts = describer.prompts.with_tags(&["joy", "surprised", "sad"]);
        describer.max_history_size = 3;

        let mut previous = None;
        for name in ["frame_0.png", "frame_1.png", "frame_0.png", "frame_1.png"] {
            let image = image::open(frames_dir.join(name)).expect("decode frame");
            let start = std::time::Instant::now();
            let comment = describer.describe(&image, previous.as_ref()).await.expect("describe");
            println!("{name} ({:.1}s): {comment}
", start.elapsed().as_secs_f64());
            previous = Some(image);
        }
    }

    #[test]
    fn session_state_override_filters_fields_and_can_be_cleared() {
        let mut session = SessionState::default();
        let given = HashMap::from([
            ("system_prompt".to_string(), "custom system".to_string()),
            ("greeting_prompt".to_string(), "custom greeting".to_string()),
            ("not_a_field".to_string(), "ignored".to_string()),
        ]);
        session.set_override(&given);

        let base = Promptset::default();
        let describer = session.describer_fields(&base, &[]);
        assert_eq!(describer.system_prompt, "custom system");
        assert_eq!(describer.greeting_prompt, base.greeting_prompt, "greeting isn't a describer field");

        let pregen = session.pregen_fields(&base, &[]);
        assert_eq!(pregen.system_prompt, "custom system");
        assert_eq!(pregen.greeting_prompt, "custom greeting");

        session.clear();
        assert_eq!(session.describer_fields(&base, &[]), base.substitute_tags(&[]));
    }

    #[test]
    fn system_message_prompts_and_emptiness_check() {
        let base = Promptset::default();
        let prompts = system_message_prompts(&base);
        assert_eq!(prompts.get("greeting"), Some(&base.greeting_prompt));
        assert!(!system_messages_are_empty(&base));

        let mut blank = base.clone();
        blank.greeting_prompt = "  ".to_string();
        blank.farewell_prompt = String::new();
        blank.lonely_prompt = "\t".to_string();
        assert!(system_messages_are_empty(&blank));
    }

    #[test]
    fn pregen_sequencer_ignores_stale_completions() {
        let sequencer = PregenSequencer::default();
        let first = sequencer.begin();
        let second = sequencer.begin();
        assert_ne!(first, second);
        assert!(!sequencer.is_current(first), "a newer run must make the older one stale");
        assert!(sequencer.is_current(second));
    }
}
