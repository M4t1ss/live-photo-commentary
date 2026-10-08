//! Named promptsets (`prompts.py`): YAML files in a directory, each
//! overriding some of the describer's prompt fields plus three
//! system-message prompts (greeting, farewell, lonely).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use vlm_describer::TAGS_PLACEHOLDER;

pub const DEFAULT_NAME: &str = "default";

const DEFAULT_GREETING_PROMPT: &str =
    "Give a single short sentence greeting to introduce yourself as a live commentary assistant.";
const DEFAULT_FAREWELL_PROMPT: &str =
    "Give a single short sentence farewell for when you are done observing.";
const DEFAULT_LONELY_PROMPT: &str =
    "Give a single short sentence expressing that you are bored waiting for the user.";

/// All eight promptset fields. Field order matches `FIELDS + SYSTEM_MESSAGE_FIELDS`
/// in Python, which is also the order `save` writes them in.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Promptset {
    pub system_prompt: String,
    pub prompt: String,
    pub first_prompt: String,
    pub history_prompt: String,
    pub compact_prompt: String,
    pub greeting_prompt: String,
    pub farewell_prompt: String,
    pub lonely_prompt: String,
}

impl Default for Promptset {
    fn default() -> Self {
        let prompts = vlm_describer::Prompts::default();
        Self {
            system_prompt: prompts.system_prompt,
            prompt: prompts.prompt,
            first_prompt: prompts.first_prompt,
            history_prompt: prompts.history_prompt,
            compact_prompt: prompts.compact_prompt,
            greeting_prompt: DEFAULT_GREETING_PROMPT.to_string(),
            farewell_prompt: DEFAULT_FAREWELL_PROMPT.to_string(),
            lonely_prompt: DEFAULT_LONELY_PROMPT.to_string(),
        }
    }
}

impl Promptset {
    /// Each field from `raw` (a YAML mapping), falling back to its default
    /// when absent (`prompts.load` in Python).
    fn from_yaml_with_defaults(raw: &serde_yaml_ng::Value) -> Promptset {
        let d = Promptset::default();
        let get = |key: &str, default: String| -> String {
            raw.get(key).and_then(|v| v.as_str()).map(str::to_string).unwrap_or(default)
        };
        Promptset {
            system_prompt: get("system_prompt", d.system_prompt),
            prompt: get("prompt", d.prompt),
            first_prompt: get("first_prompt", d.first_prompt),
            history_prompt: get("history_prompt", d.history_prompt),
            compact_prompt: get("compact_prompt", d.compact_prompt),
            greeting_prompt: get("greeting_prompt", d.greeting_prompt),
            farewell_prompt: get("farewell_prompt", d.farewell_prompt),
            lonely_prompt: get("lonely_prompt", d.lonely_prompt),
        }
    }

    /// Replaces [`TAGS_PLACEHOLDER`] in every field with the available
    /// emotion tags, written as `{tag}` marks; `neutral` is always included
    /// (`substitute_tags` in Python).
    pub fn substitute_tags(&self, tag_names: &[&str]) -> Promptset {
        let mut names: Vec<&str> = Vec::new();
        for &name in tag_names.iter().chain(&["neutral"]) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
        let tag_list = names.iter().map(|n| format!("{{{n}}}")).collect::<Vec<_>>().join(", ");
        let fill = |s: &str| s.replace(TAGS_PLACEHOLDER, &tag_list);
        Promptset {
            system_prompt: fill(&self.system_prompt),
            prompt: fill(&self.prompt),
            first_prompt: fill(&self.first_prompt),
            history_prompt: fill(&self.history_prompt),
            compact_prompt: fill(&self.compact_prompt),
            greeting_prompt: fill(&self.greeting_prompt),
            farewell_prompt: fill(&self.farewell_prompt),
            lonely_prompt: fill(&self.lonely_prompt),
        }
    }
}

impl Promptset {
    /// The five fields a [`vlm_describer::Describer`] takes, dropping the
    /// three system-message prompts (used to push the active promptset onto
    /// the live describer without a model reload; `_apply_prompt_fields` in
    /// Python).
    pub fn describer_prompts(&self) -> vlm_describer::Prompts {
        vlm_describer::Prompts {
            system_prompt: self.system_prompt.clone(),
            prompt: self.prompt.clone(),
            first_prompt: self.first_prompt.clone(),
            history_prompt: self.history_prompt.clone(),
            compact_prompt: self.compact_prompt.clone(),
        }
    }
}

/// Named promptsets stored as `*.yaml` files under `dir`.
pub struct Promptsets {
    dir: PathBuf,
}

impl Promptsets {
    pub fn new(dir: PathBuf) -> Result<Promptsets, String> {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        Ok(Promptsets { dir })
    }

    /// `default` first, then the sorted `*.yaml` stems.
    pub fn list_names(&self) -> Result<Vec<String>, String> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.dir).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_string());
                }
            }
        }
        names.sort();
        let mut result = vec![DEFAULT_NAME.to_string()];
        result.extend(names);
        Ok(result)
    }

    /// `default` or a missing file returns the defaults; otherwise each field
    /// from the file, falling back to its default.
    pub fn load(&self, name: &str) -> Result<Promptset, String> {
        if name == DEFAULT_NAME {
            return Ok(Promptset::default());
        }
        let path = self.dir.join(format!("{name}.yaml"));
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(_) => return Ok(Promptset::default()),
        };
        let raw: serde_yaml_ng::Value = serde_yaml_ng::from_str(&text).map_err(|e| e.to_string())?;
        Ok(Promptset::from_yaml_with_defaults(&raw))
    }

    /// Refuses to overwrite `default`.
    pub fn save(&self, name: &str, fields: &Promptset) -> Result<(), String> {
        if name == DEFAULT_NAME {
            return Err("Cannot overwrite the default promptset".to_string());
        }
        let path = self.dir.join(format!("{name}.yaml"));
        let yaml = serde_yaml_ng::to_string(fields).map_err(|e| e.to_string())?;
        std::fs::write(&path, yaml).map_err(|e| e.to_string())
    }

    /// Refuses to delete `default`; deleting a name that doesn't exist is not
    /// an error.
    pub fn delete(&self, name: &str) -> Result<(), String> {
        if name == DEFAULT_NAME {
            return Err("Cannot delete the default promptset".to_string());
        }
        let path = self.dir.join(format!("{name}.yaml"));
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// Moves promptset YAML files from the old Python backend's `prompts/`
/// directory to `new_dir`, if `new_dir` doesn't exist yet (one-time
/// migration, mirrored by `avatar::migrate_user_models`).
pub fn migrate(old_dir: &Path, new_dir: &Path) -> Result<(), String> {
    if new_dir.exists() || !old_dir.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(new_dir).map_err(|e| e.to_string())?;
    for entry in std::fs::read_dir(old_dir).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let dest = new_dir.join(path.file_name().expect("yaml path has a file name"));
            std::fs::rename(&path, &dest).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lpc-promptsets-test-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn default_matches_vlm_describer_and_system_message_defaults() {
        let p = Promptset::default();
        assert_eq!(p.system_prompt, vlm_describer::DEFAULT_SYSTEM_PROMPT);
        assert_eq!(p.greeting_prompt, DEFAULT_GREETING_PROMPT);
        assert_eq!(p.farewell_prompt, DEFAULT_FAREWELL_PROMPT);
        assert_eq!(p.lonely_prompt, DEFAULT_LONELY_PROMPT);
    }

    #[test]
    fn list_names_has_default_first_then_sorted_stems() {
        let dir = temp_dir("list");
        std::fs::write(dir.join("zeta.yaml"), "system_prompt: z").unwrap();
        std::fs::write(dir.join("alpha.yaml"), "system_prompt: a").unwrap();
        std::fs::write(dir.join("not-yaml.txt"), "ignored").unwrap();
        let sets = Promptsets::new(dir.clone()).unwrap();
        assert_eq!(sets.list_names().unwrap(), vec!["default", "alpha", "zeta"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_missing_file_or_default_name_returns_defaults() {
        let dir = temp_dir("missing");
        let sets = Promptsets::new(dir.clone()).unwrap();
        assert_eq!(sets.load(DEFAULT_NAME).unwrap(), Promptset::default());
        assert_eq!(sets.load("no-such-set").unwrap(), Promptset::default());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_fills_missing_fields_from_defaults() {
        let dir = temp_dir("partial");
        std::fs::write(dir.join("partial.yaml"), "system_prompt: custom system prompt\n").unwrap();
        let sets = Promptsets::new(dir.clone()).unwrap();
        let loaded = sets.load("partial").unwrap();
        assert_eq!(loaded.system_prompt, "custom system prompt");
        assert_eq!(loaded.prompt, Promptset::default().prompt);
        assert_eq!(loaded.greeting_prompt, DEFAULT_GREETING_PROMPT);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_then_load_round_trips_and_refuses_default() {
        let dir = temp_dir("save");
        let sets = Promptsets::new(dir.clone()).unwrap();
        let fields = Promptset { system_prompt: "a custom set".to_string(), ..Promptset::default() };
        sets.save("custom", &fields).unwrap();
        assert_eq!(sets.load("custom").unwrap(), fields);
        assert!(sets.save(DEFAULT_NAME, &fields).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delete_refuses_default_and_is_lenient_on_missing() {
        let dir = temp_dir("delete");
        let sets = Promptsets::new(dir.clone()).unwrap();
        assert!(sets.delete(DEFAULT_NAME).is_err());
        sets.delete("never-existed").unwrap();

        let fields = Promptset::default();
        sets.save("to-delete", &fields).unwrap();
        sets.delete("to-delete").unwrap();
        assert_eq!(sets.load("to-delete").unwrap(), Promptset::default());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn substitute_tags_dedupes_and_always_includes_neutral() {
        let fields = Promptset { system_prompt: "chosen from: <|tags|>.".to_string(), ..Promptset::default() };
        let substituted = fields.substitute_tags(&["joy", "surprised", "joy"]);
        assert_eq!(substituted.system_prompt, "chosen from: {joy}, {surprised}, {neutral}.");
    }

    #[test]
    fn describer_prompts_carries_the_five_describer_fields() {
        let fields = Promptset {
            system_prompt: "custom system".to_string(),
            greeting_prompt: "custom greeting".to_string(),
            ..Promptset::default()
        };
        let prompts = fields.describer_prompts();
        assert_eq!(prompts.system_prompt, "custom system");
        assert_eq!(prompts.prompt, Promptset::default().prompt);
        // Greeting isn't a describer field; describer_prompts has no such slot.
    }

    #[test]
    fn migrate_moves_yaml_files_once() {
        let old_dir = temp_dir("migrate-old");
        let new_dir = temp_dir("migrate-new");
        std::fs::remove_dir_all(&new_dir).ok(); // only create_dir_all'd by temp_dir(); must not exist yet
        std::fs::write(old_dir.join("japanese.yaml"), "system_prompt: nihongo").unwrap();

        migrate(&old_dir, &new_dir).unwrap();
        assert!(new_dir.join("japanese.yaml").exists());
        assert!(!old_dir.join("japanese.yaml").exists());

        // Second call is a no-op since new_dir now exists.
        std::fs::write(old_dir.join("other.yaml"), "system_prompt: other").unwrap();
        migrate(&old_dir, &new_dir).unwrap();
        assert!(!new_dir.join("other.yaml").exists());

        std::fs::remove_dir_all(&old_dir).ok();
        std::fs::remove_dir_all(&new_dir).ok();
    }

    #[test]
    fn migrates_the_real_backend_prompts_dir_if_present() {
        let real_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("backend/prompts");
        if !real_dir.exists() {
            eprintln!("skipping: {} not present on this machine", real_dir.display());
            return;
        }
        for entry in std::fs::read_dir(&real_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
                let text = std::fs::read_to_string(&path).unwrap();
                let raw: serde_yaml_ng::Value = serde_yaml_ng::from_str(&text).unwrap();
                Promptset::from_yaml_with_defaults(&raw);
            }
        }
    }
}
