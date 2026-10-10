//! Application settings, persisted to `settings.json`, with a one-time
//! migration from the old Python backend's `.env`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Fields masked as `"***"` by [`Config::masked`] when set.
const API_KEY_FIELDS: [&str; 3] =
    ["gemini_api_key", "openai_api_key", "openai_compat_api_key"];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub vlm_provider: Option<String>,
    pub vlm_model: Option<String>,
    pub vlm_model_overrides: Option<String>,
    pub tts_voice: String,
    pub pre_screenshot_delay: f64,
    pub difference_threshold: f64,
    pub difference_measure: String,
    pub max_history_size: u32,
    pub gemini_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub openai_base_url: Option<String>,
    pub openai_compat_api_key: Option<String>,
    pub active_promptset: String,
    pub active_model: String,
    pub idle_to_dance_secs: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vlm_provider: None,
            vlm_model: None,
            vlm_model_overrides: None,
            tts_voice: "af_heart".to_string(),
            pre_screenshot_delay: 2.0,
            difference_threshold: 0.0,
            difference_measure: "mse".to_string(),
            max_history_size: 0,
            gemini_api_key: None,
            openai_api_key: None,
            openai_base_url: None,
            openai_compat_api_key: None,
            active_promptset: "default".to_string(),
            active_model: "default".to_string(),
            idle_to_dance_secs: 180,
        }
    }
}

impl Config {
    /// Loads `settings.json` if it exists; otherwise migrates it once from the
    /// old backend's `.env` (if that exists), persisting the result; otherwise
    /// returns the defaults without writing anything.
    pub fn load_or_migrate(settings_path: &Path, env_path: &Path) -> Result<Config, String> {
        if settings_path.exists() {
            return Config::load(settings_path);
        }
        if env_path.exists() {
            let config = Config::from_env_file(env_path)?;
            config.save(settings_path)?;
            return Ok(config);
        }
        Ok(Config::default())
    }

    pub fn load(path: &Path) -> Result<Config, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }

    /// Writes `settings.json` atomically (temp file + rename).
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Applies `updates` on top of `self`, keeping only fields `Config` has;
    /// a JSON `null` sets an `Option` field back to `None`.
    pub fn apply(&self, updates: &serde_json::Map<String, serde_json::Value>) -> Result<Config, String> {
        let mut value = serde_json::to_value(self).map_err(|e| e.to_string())?;
        let obj = value.as_object_mut().expect("Config should serialize to a JSON object");
        for (key, update) in updates {
            if obj.contains_key(key) {
                obj.insert(key.clone(), update.clone());
            }
        }
        serde_json::from_value(value).map_err(|e| e.to_string())
    }

    /// A copy with the API key fields replaced by `"***"` where set (what the
    /// `config` event and `get_config` send).
    pub fn masked(&self) -> Config {
        let mut masked = self.clone();
        for field in API_KEY_FIELDS {
            let slot = match field {
                "gemini_api_key" => &mut masked.gemini_api_key,
                "openai_api_key" => &mut masked.openai_api_key,
                "openai_compat_api_key" => &mut masked.openai_compat_api_key,
                _ => unreachable!(),
            };
            if slot.is_some() {
                *slot = Some("***".to_string());
            }
        }
        masked
    }

    fn from_env_file(path: &Path) -> Result<Config, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Ok(Config::from_env_map(&parse_env_file(&text)))
    }

    fn from_env_map(map: &HashMap<String, String>) -> Config {
        let d = Config::default();
        let get = |key: &str| map.get(key).map(String::as_str);
        let parse_or = |key: &str, default: f64| get(key).and_then(|s| s.parse().ok()).unwrap_or(default);
        let parse_or_u32 = |key: &str, default: u32| get(key).and_then(|s| s.parse().ok()).unwrap_or(default);
        Config {
            vlm_provider: get("VLM_PROVIDER").map(str::to_string).or(d.vlm_provider),
            vlm_model: get("VLM_MODEL").map(str::to_string).or(d.vlm_model),
            vlm_model_overrides: get("VLM_MODEL_OVERRIDES").map(str::to_string).or(d.vlm_model_overrides),
            tts_voice: get("TTS_VOICE").map(str::to_string).unwrap_or(d.tts_voice),
            pre_screenshot_delay: parse_or("PRE_SCREENSHOT_DELAY", d.pre_screenshot_delay),
            difference_threshold: parse_or("DIFFERENCE_THRESHOLD", d.difference_threshold),
            difference_measure: get("DIFFERENCE_MEASURE").map(str::to_string).unwrap_or(d.difference_measure),
            max_history_size: parse_or_u32("MAX_HISTORY_SIZE", d.max_history_size),
            gemini_api_key: get("GEMINI_API_KEY").map(str::to_string).or(d.gemini_api_key),
            openai_api_key: get("OPENAI_API_KEY").map(str::to_string).or(d.openai_api_key),
            openai_base_url: get("OPENAI_BASE_URL").map(str::to_string).or(d.openai_base_url),
            openai_compat_api_key: get("OPENAI_COMPAT_API_KEY").map(str::to_string).or(d.openai_compat_api_key),
            active_promptset: get("ACTIVE_PROMPTSET").map(str::to_string).unwrap_or(d.active_promptset),
            active_model: get("ACTIVE_MODEL").map(str::to_string).unwrap_or(d.active_model),
            idle_to_dance_secs: parse_or_u32("IDLE_TO_DANCE_SECS", d.idle_to_dance_secs),
        }
        // MODEL_DIR is intentionally ignored: Rust resolves the bundled models
        // directory itself (see avatar.rs), so the field doesn't exist here.
    }
}

/// Parses `FIELD_NAME=value` lines (blank and `#`-comment lines ignored),
/// stripping one layer of matching quotes from the value as python-dotenv
/// does. Keys are upper-cased to match `Config`'s env var names.
fn parse_env_file(text: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else { continue };
        let mut value = value.trim();
        if value.len() >= 2 {
            let bytes = value.as_bytes();
            let quoted = (bytes[0] == b'"' && bytes[value.len() - 1] == b'"')
                || (bytes[0] == b'\'' && bytes[value.len() - 1] == b'\'');
            if quoted {
                value = &value[1..value.len() - 1];
            }
        }
        map.insert(key.trim().to_ascii_uppercase(), value.to_string());
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let d = Config::default();
        assert_eq!(d.vlm_provider, None);
        assert_eq!(d.tts_voice, "af_heart");
        assert_eq!(d.pre_screenshot_delay, 2.0);
        assert_eq!(d.difference_threshold, 0.0);
        assert_eq!(d.difference_measure, "mse");
        assert_eq!(d.max_history_size, 0);
        assert_eq!(d.active_promptset, "default");
        assert_eq!(d.active_model, "default");
        assert_eq!(d.idle_to_dance_secs, 180);
    }

    #[test]
    fn apply_ignores_unknown_fields_and_updates_known_ones() {
        let base = Config::default();
        let updates = serde_json::json!({
            "tts_voice": "af_nicole",
            "idle_to_dance_secs": 60,
            "not_a_real_field": "ignored",
        });
        let updated = base.apply(updates.as_object().unwrap()).unwrap();
        assert_eq!(updated.tts_voice, "af_nicole");
        assert_eq!(updated.idle_to_dance_secs, 60);
        assert_eq!(updated.vlm_provider, None);
    }

    #[test]
    fn apply_null_clears_an_option_field() {
        let base = Config { gemini_api_key: Some("secret".to_string()), ..Config::default() };
        let updates = serde_json::json!({ "gemini_api_key": null });
        let updated = base.apply(updates.as_object().unwrap()).unwrap();
        assert_eq!(updated.gemini_api_key, None);
    }

    #[test]
    fn masked_only_replaces_set_api_keys() {
        let c = Config { gemini_api_key: Some("secret".to_string()), ..Config::default() };
        let masked = c.masked();
        assert_eq!(masked.gemini_api_key, Some("***".to_string()));
        assert_eq!(masked.openai_api_key, None);
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = std::env::temp_dir().join(format!("lpc-config-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        let c = Config { tts_voice: "af_nicole".to_string(), ..Config::default() };
        c.save(&path).unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(c, loaded);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_or_migrate_returns_defaults_without_writing_when_neither_file_exists() {
        let dir = std::env::temp_dir().join(format!("lpc-config-test-fresh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let settings_path = dir.join("settings.json");
        let env_path = dir.join(".env");
        let config = Config::load_or_migrate(&settings_path, &env_path).unwrap();
        assert_eq!(config, Config::default());
        assert!(!settings_path.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_or_migrate_parses_env_file_and_persists_it() {
        let dir = std::env::temp_dir().join(format!("lpc-config-test-migrate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let settings_path = dir.join("settings.json");
        let env_path = dir.join(".env");
        std::fs::write(
            &env_path,
            "VLM_PROVIDER=local\n\
             VLM_MODEL=apple/FastVLM-0.5B\n\
             TTS_VOICE=af_heart\n\
             PRE_SCREENSHOT_DELAY=2\n\
             DIFFERENCE_THRESHOLD=0\n\
             DIFFERENCE_MEASURE=mse\n\
             MAX_HISTORY_SIZE=0\n\
             MODEL_DIR=..\\models\n\
             ACTIVE_PROMPTSET=default\n\
             ACTIVE_MODEL=default\n\
             IDLE_TO_DANCE_SECS=10\n",
        )
        .unwrap();

        let config = Config::load_or_migrate(&settings_path, &env_path).unwrap();
        assert_eq!(config.vlm_provider, Some("local".to_string()));
        assert_eq!(config.vlm_model, Some("apple/FastVLM-0.5B".to_string()));
        assert_eq!(config.idle_to_dance_secs, 10);
        assert!(settings_path.exists(), "migration should persist settings.json");

        // A second load now reads settings.json back, unaffected by .env.
        std::fs::remove_file(&env_path).unwrap();
        let reloaded = Config::load_or_migrate(&settings_path, &env_path).unwrap();
        assert_eq!(reloaded, config);

        std::fs::remove_dir_all(&dir).ok();
    }
}
