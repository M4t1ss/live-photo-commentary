//! Avatar models (`models.py`) and the camelCase `/model-config` mapping
//! (`main.py`): bundled and user-uploaded GLB files, each optionally paired
//! with a YAML configuration file (bones, visemes, emotion tags, animations).

use std::path::{Path, PathBuf};

pub const DEFAULT_NAME: &str = "default";

/// Bundled models under `model_dir`, plus user-uploaded ones under `user_dir`.
pub struct AvatarModels {
    model_dir: PathBuf,
    user_dir: PathBuf,
}

impl AvatarModels {
    pub fn new(model_dir: PathBuf, user_dir: PathBuf) -> Result<AvatarModels, String> {
        std::fs::create_dir_all(&user_dir).map_err(|e| e.to_string())?;
        Ok(AvatarModels { model_dir, user_dir })
    }

    pub fn user_dir(&self) -> &Path {
        &self.user_dir
    }

    /// `default`, then bundled `*.glb` stems (except `model`, the active
    /// model's copy) sorted, then `user/<stem>` for user models, sorted.
    pub fn list_names(&self) -> Vec<String> {
        let mut result = vec![DEFAULT_NAME.to_string()];
        result.extend(glb_stems(&self.model_dir).into_iter().filter(|stem| stem != "model"));
        result.extend(glb_stems(&self.user_dir).into_iter().map(|stem| format!("user/{stem}")));
        result
    }

    /// Returns `(glb_path, yaml_path)`; `yaml_path` falls back to
    /// `model_dir/model.yaml` when `name`'s own YAML file doesn't exist.
    pub fn resolve(&self, name: &str) -> (PathBuf, PathBuf) {
        let fallback_yaml = self.model_dir.join("model.yaml");
        if name.is_empty() || name == DEFAULT_NAME {
            return (self.model_dir.join("model.glb"), fallback_yaml);
        }
        if let Some(stem) = name.strip_prefix("user/") {
            let yaml_path = self.user_dir.join(format!("{stem}.yaml"));
            let yaml_path = if yaml_path.exists() { yaml_path } else { fallback_yaml };
            return (self.user_dir.join(format!("{stem}.glb")), yaml_path);
        }
        let yaml_path = self.model_dir.join(format!("{name}.yaml"));
        let yaml_path = if yaml_path.exists() { yaml_path } else { fallback_yaml };
        (self.model_dir.join(format!("{name}.glb")), yaml_path)
    }

    /// Stores a GLB/VRM file in the user models directory, sanitizing its
    /// stem to `[A-Za-z0-9_-]`. Returns the model name (`user/<stem>`).
    pub fn save_user_model(&self, filename: &str, data: &[u8]) -> Result<String, String> {
        std::fs::create_dir_all(&self.user_dir).map_err(|e| e.to_string())?;
        let stem = Path::new(filename).file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        let stem = sanitize_stem(stem);
        std::fs::write(self.user_dir.join(format!("{stem}.glb")), data).map_err(|e| e.to_string())?;
        Ok(format!("user/{stem}"))
    }

    /// The keys of the resolved YAML's `tags` mapping; `[]` if there is none.
    pub fn tag_names(&self, name: &str) -> Result<Vec<String>, String> {
        let (_, yaml_path) = self.resolve(name);
        if !yaml_path.exists() {
            return Ok(Vec::new());
        }
        let raw = load_merged_yaml(&yaml_path)?;
        Ok(raw
            .get("tags")
            .and_then(|t| t.as_object())
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default())
    }

    /// The camelCase mapping the frontend consumes (`/model-config` in
    /// Python); `{}` if `name` has no YAML file.
    pub fn model_config(&self, name: &str) -> Result<serde_json::Value, String> {
        let (_, yaml_path) = self.resolve(name);
        if !yaml_path.exists() {
            return Ok(serde_json::json!({}));
        }
        let raw = load_merged_yaml(&yaml_path)?;
        Ok(build_model_config(&raw))
    }
}

/// Moves user model files from the old Python backend's `user_models/`
/// directory to `new_dir`, if `new_dir` doesn't exist yet (one-time
/// migration, mirrored by `promptsets::migrate`).
pub fn migrate_user_models(old_dir: &Path, new_dir: &Path) -> Result<(), String> {
    if new_dir.exists() || !old_dir.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(new_dir).map_err(|e| e.to_string())?;
    for entry in std::fs::read_dir(old_dir).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_file() {
            let dest = new_dir.join(path.file_name().expect("file should have a name"));
            std::fs::rename(&path, &dest).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Sorted stems of the `*.glb` files directly inside `dir` (`[]` if `dir`
/// doesn't exist).
fn glb_stems(dir: &Path) -> Vec<String> {
    let mut stems: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            if path.extension().and_then(|e| e.to_str()) != Some("glb") {
                return None;
            }
            path.file_stem().and_then(|s| s.to_str()).map(str::to_string)
        })
        .collect();
    stems.sort();
    stems
}

/// Replaces every character outside `[A-Za-z0-9_-]` with `_`
/// (`re.sub(r"[^\w\-]", "_", stem, flags=re.ASCII)` in Python).
fn sanitize_stem(stem: &str) -> String {
    stem.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

/// Parses a YAML file and resolves its `<<` merge keys (used by `viseme_map`
/// to reuse base shapes), returning the result as JSON.
fn load_merged_yaml(path: &Path) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let raw: serde_yaml_ng::Value = serde_yaml_ng::from_str(&text).map_err(|e| e.to_string())?;
    let merged = resolve_merge_keys(raw);
    serde_json::to_value(&merged).map_err(|e| e.to_string())
}

/// Resolves YAML's `<<` merge keys bottom-up. `serde_yaml_ng` expands
/// anchors/aliases on parse but, like most YAML libraries, leaves `<<` as a
/// literal mapping key, which the frontend would otherwise see.
///
/// Per the merge key spec: a mapping's own keys always win; of the merge
/// sources (its mapping, or each mapping in its sequence), an earlier source
/// wins over a later one for the same key.
fn resolve_merge_keys(value: serde_yaml_ng::Value) -> serde_yaml_ng::Value {
    use serde_yaml_ng::Value;
    match value {
        Value::Sequence(seq) => Value::Sequence(seq.into_iter().map(resolve_merge_keys).collect()),
        Value::Mapping(map) => {
            let merge_key = Value::String("<<".to_string());
            let mut own = serde_yaml_ng::Mapping::new();
            let mut merge_sources: Option<Value> = None;
            for (k, v) in map {
                let v = resolve_merge_keys(v);
                if k == merge_key {
                    merge_sources = Some(v);
                } else {
                    own.insert(k, v);
                }
            }

            let mut result = serde_yaml_ng::Mapping::new();
            if let Some(sources) = merge_sources {
                let sources = match sources {
                    Value::Sequence(seq) => seq,
                    other => vec![other],
                };
                for source in sources {
                    if let Value::Mapping(source_map) = source {
                        for (k, v) in source_map {
                            result.entry(k).or_insert(v);
                        }
                    }
                }
            }
            for (k, v) in own {
                result.insert(k, v); // a mapping's own keys always win over its merge sources
            }
            Value::Mapping(result)
        }
        other => other,
    }
}

/// Builds the camelCase JSON `model_config` sends to the frontend from a
/// parsed, merge-key-resolved avatar YAML (Appendix B in RUSTIFICATION.md).
fn build_model_config(raw: &serde_json::Value) -> serde_json::Value {
    let get = |key: &str| raw.get(key).cloned().unwrap_or(serde_json::Value::Null);
    let get_or = |key: &str, default: serde_json::Value| raw.get(key).cloned().unwrap_or(default);
    let animations = raw.get("animations").cloned().unwrap_or_else(|| serde_json::json!({}));
    let anim = |key: &str| animations.get(key).cloned().unwrap_or(serde_json::Value::Null);

    serde_json::json!({
        "jawBone": get("jaw_bone"),
        "jawAxis": get_or("jaw_axis", serde_json::json!("x")),
        "minJawAngle": get_or("min_jaw_angle", serde_json::json!(0.0)),
        "maxJawAngle": get_or("max_jaw_angle", serde_json::json!(0.15)),
        "neckBone": get("neck_bone"),
        "maxNeckAngle": get_or("max_neck_angle", serde_json::json!(0.3)),
        "headBone": get("head_bone"),
        "maxHeadAngle": get_or("max_head_angle", serde_json::json!(0.5)),
        "leftEyeBone": get("left_eye_bone"),
        "rightEyeBone": get("right_eye_bone"),
        "maxEyeAngle": get_or("max_eye_angle", serde_json::json!(0.4)),
        "maxEnvelopeDuration": get_or("max_envelope_duration", serde_json::json!(0.3)),
        "visemeMap": get_or("viseme_map", serde_json::json!({})),
        "blink": get("blink"),
        "tags": get_or("tags", serde_json::json!({})),
        "walkInAnimation": anim("walk_in"),
        "walkOutAnimation": anim("walk_out"),
        "helloAnimation": anim("hello"),
        "goodbyeAnimation": anim("goodbye"),
        "idleAnimation": anim("idle"),
        "lonelyAnimation": anim("lonely"),
        "dancingAnimation": anim("dancing"),
        "talkingAnimation": anim("talking"),
        "gazeSpeed": get_or("gaze_speed", serde_json::json!(4.0)),
        "gazeSuspension": get("gaze_suspension"),
        "lighting": get("lighting"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dirs(name: &str) -> (PathBuf, PathBuf) {
        let base = std::env::temp_dir().join(format!("lpc-avatar-test-{name}-{}", std::process::id()));
        let model_dir = base.join("models");
        let user_dir = base.join("user_models");
        std::fs::create_dir_all(&model_dir).unwrap();
        (model_dir, user_dir)
    }

    #[test]
    fn list_names_excludes_the_model_stem_and_sorts_each_group() {
        let (model_dir, user_dir) = temp_dirs("list");
        for name in ["model", "zeta", "alpha"] {
            std::fs::write(model_dir.join(format!("{name}.glb")), b"").unwrap();
        }
        std::fs::create_dir_all(&user_dir).unwrap();
        for name in ["bravo", "charlie"] {
            std::fs::write(user_dir.join(format!("{name}.glb")), b"").unwrap();
        }
        let models = AvatarModels::new(model_dir.clone(), user_dir.clone()).unwrap();
        assert_eq!(
            models.list_names(),
            vec!["default", "alpha", "zeta", "user/bravo", "user/charlie"]
        );
        std::fs::remove_dir_all(model_dir.parent().unwrap()).ok();
    }

    #[test]
    fn resolve_falls_back_to_the_bundled_model_yaml() {
        let (model_dir, user_dir) = temp_dirs("resolve");
        std::fs::write(model_dir.join("model.yaml"), "neck_bone: fallback").unwrap();
        std::fs::write(model_dir.join("named.glb"), b"").unwrap();
        std::fs::write(model_dir.join("named.yaml"), "neck_bone: own").unwrap();
        let models = AvatarModels::new(model_dir.clone(), user_dir.clone()).unwrap();

        let (glb, yaml) = models.resolve(DEFAULT_NAME);
        assert_eq!(glb, model_dir.join("model.glb"));
        assert_eq!(yaml, model_dir.join("model.yaml"));

        // Has its own YAML.
        let (glb, yaml) = models.resolve("named");
        assert_eq!(glb, model_dir.join("named.glb"));
        assert_eq!(yaml, model_dir.join("named.yaml"));

        // No YAML of its own: falls back to model_dir/model.yaml.
        let (glb, yaml) = models.resolve("no-yaml");
        assert_eq!(glb, model_dir.join("no-yaml.glb"));
        assert_eq!(yaml, model_dir.join("model.yaml"));

        // User model, with its own YAML.
        std::fs::create_dir_all(&user_dir).unwrap();
        std::fs::write(user_dir.join("mine.yaml"), "neck_bone: mine").unwrap();
        let (glb, yaml) = models.resolve("user/mine");
        assert_eq!(glb, user_dir.join("mine.glb"));
        assert_eq!(yaml, user_dir.join("mine.yaml"));

        // User model without its own YAML: falls back to model_dir/model.yaml.
        let (glb, yaml) = models.resolve("user/no-yaml");
        assert_eq!(glb, user_dir.join("no-yaml.glb"));
        assert_eq!(yaml, model_dir.join("model.yaml"));

        std::fs::remove_dir_all(model_dir.parent().unwrap()).ok();
    }

    #[test]
    fn save_user_model_sanitizes_the_stem_and_returns_its_name() {
        let (model_dir, user_dir) = temp_dirs("save");
        let models = AvatarModels::new(model_dir.clone(), user_dir.clone()).unwrap();
        let name = models.save_user_model("My Model! (v2).glb", b"glb-bytes").unwrap();
        assert_eq!(name, "user/My_Model___v2_");
        assert_eq!(std::fs::read(user_dir.join("My_Model___v2_.glb")).unwrap(), b"glb-bytes");
        std::fs::remove_dir_all(model_dir.parent().unwrap()).ok();
    }

    #[test]
    fn tag_names_and_model_config_are_empty_without_a_yaml_file() {
        let (model_dir, user_dir) = temp_dirs("empty");
        let models = AvatarModels::new(model_dir.clone(), user_dir.clone()).unwrap();
        assert_eq!(models.tag_names(DEFAULT_NAME).unwrap(), Vec::<String>::new());
        assert_eq!(models.model_config(DEFAULT_NAME).unwrap(), serde_json::json!({}));
        std::fs::remove_dir_all(model_dir.parent().unwrap()).ok();
    }

    #[test]
    fn model_config_resolves_merge_keys_fills_defaults_and_reads_tags() {
        let (model_dir, user_dir) = temp_dirs("config");
        std::fs::write(
            model_dir.join("model.yaml"),
            "neck_bone: J_Neck\n\
             max_neck_angle: 0.3\n\
             bases:\n\
             \x20 close_rounded: &close_rounded\n\
             \x20   Fcl_MTH_U: 0.85\n\
             viseme_map:\n\
             \x20 u:\n\
             \x20   <<: *close_rounded\n\
             \x20   Fcl_MTH_U: 0.90\n\
             tags:\n\
             \x20 neutral: {}\n\
             \x20 joy: { Fcl_BRW_Joy: 1.0 }\n\
             animations:\n\
             \x20 idle: Idle.vrma\n",
        )
        .unwrap();
        let models = AvatarModels::new(model_dir.clone(), user_dir.clone()).unwrap();

        let mut tags = models.tag_names(DEFAULT_NAME).unwrap();
        tags.sort();
        assert_eq!(tags, vec!["joy", "neutral"]);

        let config = models.model_config(DEFAULT_NAME).unwrap();
        assert_eq!(config["neckBone"], serde_json::json!("J_Neck"));
        assert_eq!(config["maxNeckAngle"], serde_json::json!(0.3));
        assert_eq!(config["jawBone"], serde_json::Value::Null);
        assert_eq!(config["maxJawAngle"], serde_json::json!(0.15)); // default
        assert_eq!(config["visemeMap"]["u"], serde_json::json!({ "Fcl_MTH_U": 0.90 }));
        assert_eq!(config["idleAnimation"], serde_json::json!("Idle.vrma"));
        assert_eq!(config["walkInAnimation"], serde_json::Value::Null);
        assert_eq!(config["gazeSpeed"], serde_json::json!(4.0)); // default

        std::fs::remove_dir_all(model_dir.parent().unwrap()).ok();
    }

    #[test]
    fn sanitize_stem_keeps_word_chars_and_hyphens() {
        assert_eq!(sanitize_stem("a b-c_d!e"), "a_b-c_d_e");
    }

    fn yaml(text: &str) -> serde_yaml_ng::Value {
        serde_yaml_ng::from_str(text).unwrap()
    }

    #[test]
    fn merge_keys_single_source_lets_own_keys_win() {
        // Spec example: a mapping's own keys override its merge source, but
        // keys the source has and the mapping doesn't are kept.
        let merged = resolve_merge_keys(yaml(
            "ref: &ref\n\
             \x20 merged_key: merged\n\
             \x20 added_key: merged\n\
             dict:\n\
             \x20 <<: *ref\n\
             \x20 top_key: given\n\
             \x20 merged_key: given\n",
        ));
        assert_eq!(
            merged["dict"],
            yaml("top_key: given\nmerged_key: given\nadded_key: merged\n")
        );
    }

    #[test]
    fn merge_keys_list_of_sources_resolves_in_first_wins_order() {
        // Spec example ("Override"): of several merge sources, the first in
        // the list wins over later ones for the same key; the mapping's own
        // keys win over all of them.
        let merged = resolve_merge_keys(yaml(
            "center: &CENTER { x: 1, y: 2 }\n\
             left: &LEFT { x: 0, y: 2 }\n\
             big: &BIG { r: 10 }\n\
             small: &SMALL { r: 1 }\n\
             result:\n\
             \x20 <<: [*BIG, *LEFT, *SMALL]\n\
             \x20 x: 1\n\
             \x20 label: center/big\n",
        ));
        assert_eq!(merged["result"], yaml("{ x: 1, y: 2, r: 10, label: center/big }"));
    }

    #[test]
    fn merge_keys_resolve_inside_nested_sequences() {
        let merged = resolve_merge_keys(yaml(
            "base: &base { a: 1 }\n\
             list:\n\
             \x20 - <<: *base\n\
             \x20   b: 2\n",
        ));
        assert_eq!(merged["list"][0], yaml("{ a: 1, b: 2 }"));
    }

    #[test]
    fn migrate_user_models_moves_files_once() {
        let old_dir = std::env::temp_dir().join(format!("lpc-avatar-migrate-old-{}", std::process::id()));
        let new_dir = std::env::temp_dir().join(format!("lpc-avatar-migrate-new-{}", std::process::id()));
        std::fs::remove_dir_all(&old_dir).ok();
        std::fs::remove_dir_all(&new_dir).ok();
        std::fs::create_dir_all(&old_dir).unwrap();
        std::fs::write(old_dir.join("mine.glb"), b"glb").unwrap();
        std::fs::write(old_dir.join("mine.yaml"), "neck_bone: mine").unwrap();

        migrate_user_models(&old_dir, &new_dir).unwrap();
        assert!(new_dir.join("mine.glb").exists());
        assert!(new_dir.join("mine.yaml").exists());
        assert!(!old_dir.join("mine.glb").exists());

        std::fs::write(old_dir.join("other.glb"), b"glb").unwrap();
        migrate_user_models(&old_dir, &new_dir).unwrap(); // no-op: new_dir exists
        assert!(!new_dir.join("other.glb").exists());

        std::fs::remove_dir_all(&old_dir).ok();
        std::fs::remove_dir_all(&new_dir).ok();
    }

    #[test]
    fn loads_the_real_bundled_model_yaml_without_errors() {
        let real_model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("models");
        if !real_model_dir.join("model.yaml").exists() {
            eprintln!("skipping: {} not present on this machine", real_model_dir.display());
            return;
        }
        let models = AvatarModels::new(real_model_dir, std::env::temp_dir().join("lpc-avatar-unused")).unwrap();
        let config = models.model_config(DEFAULT_NAME).unwrap();
        // The repo's model.yaml sets these explicitly; see models/model.yaml.
        assert_eq!(config["neckBone"], serde_json::json!("J_Bip_C_Neck"));
        assert_eq!(config["maxEyeAngle"], serde_json::json!(0.15));
        assert_eq!(config["gazeSpeed"], serde_json::json!(4.0)); // not set: default
        assert!(config["visemeMap"]["u"]["Fcl_MTH_U"].as_f64().unwrap() > 0.0);
        let tags = models.tag_names(DEFAULT_NAME).unwrap();
        assert!(tags.contains(&"neutral".to_string()));
    }
}
