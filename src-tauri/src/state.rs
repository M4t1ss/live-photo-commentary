//! Session state and model lifecycle (`main.py`'s module-level globals and
//! websocket handler): which reload a `set_config` call should trigger, the
//! ad-hoc prompt overrides from the Settings "OK" button, and the
//! pre-generation sequence counter.
//!
//! Cloud describer construction (`_make_describer`'s gemini/openai
//! branches) is pure and fully ported here. The `local` branch needs the
//! catalogue rework of phase 6 (GGUF/mmproj file names per entry) plus a
//! download step, so it isn't wired up yet; `build_describer_backend`
//! reports that clearly instead of guessing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use kokoro_timestamped::{Voice, VoiceBlendPart};

use crate::config::Config;
use crate::promptsets::Promptset;

/// The five fields pushed onto the live describer (`prompts.FIELDS` in
/// Python).
const DESCRIBER_FIELDS: [&str; 5] = ["system_prompt", "prompt", "first_prompt", "history_prompt", "compact_prompt"];
/// The three system-message prompts (`prompts.SYSTEM_MESSAGE_FIELDS`).
const SYSTEM_MESSAGE_FIELDS: [&str; 3] = ["greeting_prompt", "farewell_prompt", "lonely_prompt"];

/// Which models a `set_config` call should reload (the websocket handler's
/// `set_config` case in Python).
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
/// `{voice: weight}` for a blend (`_make_synthesizer` in Python).
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

/// Loads the Kokoro synthesizer for `cfg.tts_voice` (`_make_synthesizer` in
/// Python; always model `"model"`, speed 1.0, language from the voice name).
pub async fn build_synthesizer(cfg: &Config) -> Result<kokoro_timestamped::KokoroSynthesizer, String> {
    kokoro_timestamped::KokoroSynthesizer::new(parse_voice(&cfg.tts_voice), "model", None, 1.0)
        .await
        .map_err(|e| e.to_string())
}

/// Builds the backend for a cloud VLM provider (`_make_describer` in
/// Python). `local` isn't implemented yet — see the module doc.
pub fn build_describer_backend(cfg: &Config) -> Result<vlm_describer::Backend, String> {
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
        Some("local") => Err("the local VLM provider isn't wired up yet (phase 6)".to_string()),
        Some(other) => Err(format!("unknown VLM provider {other:?}")),
        None => Err("no VLM provider configured".to_string()),
    }
}

/// Ad-hoc prompt overrides from the Settings "OK" button
/// (`_prompt_override`/`_sys_msg_override` in Python): applied to the live
/// describer and to pre-generation for this session only, cleared by an
/// explicit promptset load or delete.
#[derive(Default)]
pub struct SessionState {
    prompt_override: Option<HashMap<String, String>>,
    sys_msg_override: HashMap<String, String>,
}

impl SessionState {
    /// Splits `given` into the describer-field override and the
    /// system-message-field override, dropping anything else
    /// (`apply_prompts` in Python's websocket handler).
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
    /// ad-hoc override layered on top, tag-substituted (`_apply_promptset`
    /// in Python).
    pub fn describer_fields(&self, base: &Promptset, tag_names: &[&str]) -> Promptset {
        let mut fields = base.clone();
        if let Some(over) = &self.prompt_override {
            apply_fields(&mut fields, over);
        }
        fields.substitute_tags(tag_names)
    }

    /// All eight fields, merged for pre-generation (`_trigger_pregen`'s
    /// `all_fields` in Python, then tag-substituted). Python substitutes
    /// tags on `all_fields` too but builds the three message prompts from
    /// the *unsubstituted* fields, substituting only `system_prompt`
    /// afterwards; here every field is substituted uniformly before either
    /// is read, which only differs from Python if a custom greeting,
    /// farewell or lonely prompt itself contains the tag placeholder (the
    /// defaults never do).
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
/// suffix (`sys_msg_prompts` in Python's `_trigger_pregen`), for
/// [`crate::pipeline::Pipeline::pregen_system_messages`].
pub fn system_message_prompts(fields: &Promptset) -> HashMap<String, String> {
    HashMap::from([
        ("greeting".to_string(), fields.greeting_prompt.clone()),
        ("farewell".to_string(), fields.farewell_prompt.clone()),
        ("lonely".to_string(), fields.lonely_prompt.clone()),
    ])
}

/// True when none of the three system-message prompts has any text: pregen
/// should be skipped entirely and `system_messages` marked ready right away
/// (`_trigger_pregen`'s guard in Python).
pub fn system_messages_are_empty(fields: &Promptset) -> bool {
    [&fields.greeting_prompt, &fields.farewell_prompt, &fields.lonely_prompt].iter().all(|p| p.trim().is_empty())
}

/// Ignores a stale pre-generation completion racing a newer run
/// (`_pregen_seq` in Python's `_trigger_pregen`/`_on_done`).
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
    fn build_describer_backend_for_gemini_and_openai() {
        let cfg = Config {
            vlm_provider: Some("gemini".to_string()),
            vlm_model: Some("gemini-3.5-flash".to_string()),
            gemini_api_key: Some("secret".to_string()),
            ..Config::default()
        };
        assert!(matches!(build_describer_backend(&cfg), Ok(vlm_describer::Backend::Gemini(_))));

        let cfg = Config {
            vlm_provider: Some("openai".to_string()),
            vlm_model: Some("gpt-4o".to_string()),
            ..Config::default()
        };
        assert!(matches!(build_describer_backend(&cfg), Ok(vlm_describer::Backend::OpenAi(_))));
    }

    #[test]
    fn build_describer_backend_reports_local_and_missing_config_clearly() {
        let cfg = Config {
            vlm_provider: Some("local".to_string()),
            vlm_model: Some("org/model".to_string()),
            ..Config::default()
        };
        assert!(build_describer_backend(&cfg).is_err());

        assert!(build_describer_backend(&Config::default()).is_err());
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
