//! The VLM model catalogue (`vlm_models.yaml`). Cloud entries are model names;
//! local entries also say where their llama.cpp GGUF files are on Hugging
//! Face. The "Model overrides" setting is merged over an entry
//! (`entry_for_model` and `with_overrides`).

use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::path::Path;
use vlm_describer::Sampling;

/// Where a local model's files are on Hugging Face.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufFiles {
    pub repo: String,
    pub model_file: String,
    /// The vision projector.
    pub mmproj_file: String,
}

/// A catalogue entry as the `models` event sends it. The frontend copies
/// `generation_kwargs`, `response_re` and `base_url` into the settings; the
/// files stay in the backend.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct CatalogueEntry {
    pub name: String,
    pub display_name: String,
    pub generation_kwargs: serde_json::Map<String, serde_json::Value>,
    pub response_re: Option<String>,
    pub base_url: Option<String>,
    /// Local models only.
    #[serde(skip)]
    pub gguf: Option<GgufFiles>,
}

/// Providers in the YAML file's order, each with its entries in order
/// (a plain `Vec` keeps that order, unlike a sorted map).
#[derive(Debug)]
pub struct Catalogue {
    providers: Vec<(String, Vec<CatalogueEntry>)>,
}

impl Catalogue {
    #[cfg(test)]
    pub fn load(path: &Path) -> Result<Catalogue, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Catalogue::parse(&text)
    }

    pub fn parse(yaml_text: &str) -> Result<Catalogue, String> {
        let raw: serde_yaml_ng::Value = serde_yaml_ng::from_str(yaml_text).map_err(|e| e.to_string())?;
        let mapping = raw.as_mapping().ok_or("catalogue YAML must be a mapping of provider to entries")?;
        let mut providers = Vec::new();
        for (key, value) in mapping {
            let provider = key.as_str().ok_or("catalogue provider keys must be strings")?.to_string();
            let entries_raw =
                value.as_sequence().ok_or_else(|| format!("provider '{provider}' must be a list of entries"))?;
            let entries = entries_raw.iter().map(parse_entry).collect::<Result<Vec<_>, _>>()?;
            if provider == "local" {
                if let Some(entry) = entries.iter().find(|e| e.gguf.is_none()) {
                    return Err(format!(
                        "local model '{}' needs gguf_repo, model_file and mmproj_file",
                        entry.name
                    ));
                }
            }
            providers.push((provider, entries));
        }
        Ok(Catalogue { providers })
    }

    /// Providers in file order, each with its entries in file order.
    pub fn providers(&self) -> &[(String, Vec<CatalogueEntry>)] {
        &self.providers
    }

    /// The first entry named `model_id` across all providers; a bare
    /// `{name, display_name: name}` if there's no such entry.
    pub fn entry_for_model(&self, model_id: &str) -> CatalogueEntry {
        for (_, entries) in &self.providers {
            if let Some(entry) = entries.iter().find(|e| e.name == model_id) {
                return entry.clone();
            }
        }
        CatalogueEntry { name: model_id.to_string(), display_name: model_id.to_string(), ..Default::default() }
    }
}

impl CatalogueEntry {
    /// Merges the "Model overrides" setting (a JSON object) over the entry:
    /// `generation_kwargs` key by key, `response_re` and the three GGUF
    /// fields (`gguf_repo`, `model_file`, `mmproj_file`) as given. The
    /// PyTorch-era `processor_kwargs` and `model_kwargs` are ignored with a
    /// warning, and so is JSON that doesn't parse (as in Python).
    pub fn with_overrides(mut self, raw: &str) -> CatalogueEntry {
        let overrides = match serde_json::from_str::<serde_json::Value>(raw) {
            Ok(serde_json::Value::Object(map)) => map,
            _ => {
                log::warn!("Invalid vlm_model_overrides JSON, ignoring");
                return self;
            }
        };
        for key in ["processor_kwargs", "model_kwargs"] {
            if overrides.contains_key(key) {
                log::warn!("vlm_model_overrides: {key} only applied to the PyTorch models and is ignored");
            }
        }
        if let Some(serde_json::Value::Object(kwargs)) = overrides.get("generation_kwargs") {
            for (key, value) in kwargs {
                self.generation_kwargs.insert(key.clone(), value.clone());
            }
        }
        if let Some(value) = overrides.get("response_re") {
            self.response_re = value.as_str().map(str::to_string);
        }
        let given = |key: &str| overrides.get(key).and_then(|v| v.as_str()).map(str::to_string);
        let current = self.gguf.take();
        let pick = |key: &str, from_entry: Option<&String>| given(key).or_else(|| from_entry.cloned());
        self.gguf = match (
            pick("gguf_repo", current.as_ref().map(|g| &g.repo)),
            pick("model_file", current.as_ref().map(|g| &g.model_file)),
            pick("mmproj_file", current.as_ref().map(|g| &g.mmproj_file)),
        ) {
            (Some(repo), Some(model_file), Some(mmproj_file)) => Some(GgufFiles { repo, model_file, mmproj_file }),
            _ => None,
        };
        self
    }

    /// How a local model generates, from `generation_kwargs` (keys
    /// `max_new_tokens`, `temperature`, `top_p`, `top_k`; anything else is
    /// ignored) over `Sampling`'s defaults.
    pub fn sampling(&self) -> Sampling {
        let mut sampling = Sampling::default();
        let kwargs = &self.generation_kwargs;
        if let Some(n) = kwargs.get("max_new_tokens").and_then(|v| v.as_u64()) {
            sampling.max_new_tokens = n as usize;
        }
        if let Some(t) = kwargs.get("temperature").and_then(|v| v.as_f64()) {
            sampling.temperature = t as f32;
        }
        if let Some(p) = kwargs.get("top_p").and_then(|v| v.as_f64()) {
            sampling.top_p = Some(p as f32);
        }
        if let Some(k) = kwargs.get("top_k").and_then(|v| v.as_i64()) {
            sampling.top_k = Some(k as i32);
        }
        sampling
    }
}

/// A plain string is `{name, display_name: name}`; a mapping needs `name` and
/// takes the rest from its optional fields. A mapping with all of
/// `gguf_repo`, `model_file` and `mmproj_file` is a local model.
fn parse_entry(raw: &serde_yaml_ng::Value) -> Result<CatalogueEntry, String> {
    if let Some(name) = raw.as_str() {
        return Ok(CatalogueEntry { name: name.to_string(), display_name: name.to_string(), ..Default::default() });
    }
    let name = raw
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or("catalogue entry must be a string or a mapping with a 'name'")?
        .to_string();
    let text = |key: &str| raw.get(key).and_then(|v| v.as_str()).map(str::to_string);
    let display_name = text("display_name").unwrap_or_else(|| name.clone());
    let generation_kwargs = match raw.get("generation_kwargs") {
        None => Default::default(),
        Some(v) if v.is_null() => Default::default(),
        Some(v) => serde_json::to_value(v)
            .map_err(|e| e.to_string())?
            .as_object()
            .cloned()
            .ok_or_else(|| format!("catalogue entry '{name}': generation_kwargs must be a mapping"))?,
    };
    let gguf = match (text("gguf_repo"), text("model_file"), text("mmproj_file")) {
        (Some(repo), Some(model_file), Some(mmproj_file)) => Some(GgufFiles { repo, model_file, mmproj_file }),
        (None, None, None) => None,
        _ => return Err(format!("catalogue entry '{name}': gguf_repo, model_file and mmproj_file go together")),
    };
    Ok(CatalogueEntry {
        name,
        display_name,
        generation_kwargs,
        response_re: text("response_re"),
        base_url: text("base_url"),
        gguf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCAL_ENTRY: &str = "local:\n\
        \x20 - name: org/model\n\
        \x20   gguf_repo: org/model-GGUF\n\
        \x20   model_file: model-Q4_K_M.gguf\n\
        \x20   mmproj_file: mmproj-model.gguf\n";

    #[test]
    fn plain_string_entries_default_display_name_to_the_name() {
        let catalogue = Catalogue::parse("gemini:\n  - gemini-3.5-flash\n").unwrap();
        let providers = catalogue.providers();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].0, "gemini");
        assert_eq!(
            providers[0].1[0],
            CatalogueEntry {
                name: "gemini-3.5-flash".to_string(),
                display_name: "gemini-3.5-flash".to_string(),
                ..Default::default()
            }
        );
    }

    #[test]
    fn providers_and_entries_keep_file_order() {
        let catalogue = Catalogue::parse(
            "gemini:\n  - gemini-a\n  - gemini-b\n\
             openai:\n  - gpt-a\n",
        )
        .unwrap();
        let names: Vec<&str> = catalogue.providers().iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, vec!["gemini", "openai"]);
        assert_eq!(catalogue.providers()[0].1.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), vec!["gemini-a", "gemini-b"]);
    }

    #[test]
    fn local_entries_take_files_generation_kwargs_and_response_re() {
        let yaml = format!(
            "{LOCAL_ENTRY}\
             \x20   generation_kwargs:\n\
             \x20     temperature: 1.0\n\
             \x20     top_k: 64\n\
             \x20   response_re: '<\\|start\\|>assistant to=user<\\|message\\|>(.*?)$'\n"
        );
        let catalogue = Catalogue::parse(&yaml).unwrap();
        let entry = &catalogue.providers()[0].1[0];
        assert_eq!(entry.name, "org/model");
        assert_eq!(entry.display_name, "org/model");
        assert_eq!(
            entry.gguf,
            Some(GgufFiles {
                repo: "org/model-GGUF".to_string(),
                model_file: "model-Q4_K_M.gguf".to_string(),
                mmproj_file: "mmproj-model.gguf".to_string(),
            })
        );
        assert_eq!(entry.generation_kwargs.get("temperature"), Some(&serde_json::json!(1.0)));
        assert_eq!(entry.generation_kwargs.get("top_k"), Some(&serde_json::json!(64)));
        assert_eq!(entry.response_re.as_deref(), Some("<\\|start\\|>assistant to=user<\\|message\\|>(.*?)$"));
        assert_eq!(entry.base_url, None);
    }

    #[test]
    fn local_entries_without_files_are_rejected() {
        assert!(Catalogue::parse("local:\n  - org/model\n").unwrap_err().contains("org/model"));
        let partial = "local:\n  - name: org/model\n    gguf_repo: org/model-GGUF\n";
        assert!(Catalogue::parse(partial).unwrap_err().contains("go together"));
    }

    #[test]
    fn the_models_event_shows_no_files() {
        let catalogue = Catalogue::parse(LOCAL_ENTRY).unwrap();
        let json = serde_json::to_value(&catalogue.providers()[0].1[0]).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "name": "org/model", "display_name": "org/model", "generation_kwargs": {},
                "response_re": null, "base_url": null,
            })
        );
    }

    #[test]
    fn entry_for_model_finds_across_providers_and_falls_back_to_a_bare_entry() {
        let catalogue = Catalogue::parse(
            "gemini:\n  - name: gemini-a\n    display_name: Gemini A\n\
             local:\n  - name: org/model\n    display_name: Model\n    gguf_repo: r/m\n    model_file: m.gguf\n    mmproj_file: p.gguf\n",
        )
        .unwrap();
        assert_eq!(catalogue.entry_for_model("org/model").display_name, "Model");
        assert_eq!(catalogue.entry_for_model("gemini-a").display_name, "Gemini A");
        let fallback = catalogue.entry_for_model("unknown/model");
        assert_eq!(fallback.name, "unknown/model");
        assert_eq!(fallback.display_name, "unknown/model");
        assert!(fallback.gguf.is_none());
    }

    #[test]
    fn overrides_merge_generation_kwargs_response_re_and_files() {
        let entry = Catalogue::parse(&format!("{LOCAL_ENTRY}    generation_kwargs:\n      temperature: 0.5\n      top_k: 10\n"))
            .unwrap()
            .entry_for_model("org/model");
        let merged = entry.with_overrides(
            r#"{"generation_kwargs": {"top_k": 20, "max_new_tokens": 99}, "response_re": "(.*)",
                "model_file": "other.gguf", "processor_kwargs": {"max_pixels": 1}}"#,
        );
        assert_eq!(merged.generation_kwargs.get("temperature"), Some(&serde_json::json!(0.5)));
        assert_eq!(merged.generation_kwargs.get("top_k"), Some(&serde_json::json!(20)));
        assert_eq!(merged.generation_kwargs.get("max_new_tokens"), Some(&serde_json::json!(99)));
        assert_eq!(merged.response_re.as_deref(), Some("(.*)"));
        let gguf = merged.gguf.unwrap();
        assert_eq!((gguf.repo.as_str(), gguf.model_file.as_str(), gguf.mmproj_file.as_str()), ("org/model-GGUF", "other.gguf", "mmproj-model.gguf"));
    }

    #[test]
    fn overrides_can_supply_the_files_of_a_model_that_is_not_listed() {
        let bare = Catalogue::parse("gemini:\n  - a\n").unwrap().entry_for_model("my/model");
        let partial = bare.clone().with_overrides(r#"{"gguf_repo": "my/model-GGUF"}"#);
        assert!(partial.gguf.is_none());
        let full = bare.with_overrides(r#"{"gguf_repo": "r/m", "model_file": "m.gguf", "mmproj_file": "p.gguf"}"#);
        assert_eq!(full.gguf.unwrap().repo, "r/m");
    }

    #[test]
    fn invalid_overrides_are_ignored() {
        let entry = Catalogue::parse(LOCAL_ENTRY).unwrap().entry_for_model("org/model");
        assert_eq!(entry.clone().with_overrides("{not json"), entry);
        assert_eq!(entry.clone().with_overrides("[1, 2]"), entry);
    }

    #[test]
    fn sampling_comes_from_generation_kwargs_over_the_defaults() {
        let entry = CatalogueEntry::default();
        assert_eq!(entry.sampling(), Sampling::default());
        let entry = entry.with_overrides(
            r#"{"generation_kwargs": {"max_new_tokens": 10000, "temperature": 0.7, "top_p": 0.95, "top_k": 64, "do_sample": true}}"#,
        );
        assert_eq!(
            entry.sampling(),
            Sampling { max_new_tokens: 10000, temperature: 0.7, top_p: Some(0.95), top_k: Some(64) }
        );
    }

    #[test]
    fn loads_the_real_vlm_models_yaml_without_errors() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("vlm_models.yaml");
        let catalogue = Catalogue::load(&path).unwrap();
        let names: Vec<&str> = catalogue.providers().iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, vec!["gemini", "openai", "local"]);
        let muse = catalogue.entry_for_model("meta-models/Muse-Glimmer-30B");
        assert!(muse.response_re.is_some());
        assert_eq!(muse.sampling().max_new_tokens, 10000);
        let local = &catalogue.providers()[2].1;
        assert!(local.iter().all(|e| e.gguf.is_some()));
        let mut names: Vec<_> = local.iter().map(|e| &e.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), local.len(), "model names must be unique");
    }
}
