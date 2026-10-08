//! The VLM model catalogue (`vlm_models.yaml` plus the parsing in `main.py`).
//! Phase 1 ports today's YAML shape and its parsing into the `models` event's
//! entry shape; phase 6 replaces the local entries' fields with llama.cpp
//! GGUF/mmproj info.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// A catalogue entry as the `models` event sends it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct CatalogueEntry {
    pub name: String,
    pub display_name: String,
    pub processor_kwargs: serde_json::Map<String, serde_json::Value>,
    pub model_kwargs: serde_json::Map<String, serde_json::Value>,
    pub generation_kwargs: serde_json::Map<String, serde_json::Value>,
    pub response_re: Option<String>,
    pub base_url: Option<String>,
}

/// Providers in the YAML file's order, each with its entries in order
/// (`_VLM_CATALOGUE` in Python; a plain `Vec` keeps that order, unlike a
/// sorted map).
pub struct Catalogue {
    providers: Vec<(String, Vec<CatalogueEntry>)>,
}

impl Catalogue {
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
            providers.push((provider, entries));
        }
        Ok(Catalogue { providers })
    }

    /// Providers in file order, each with its entries in file order.
    pub fn providers(&self) -> &[(String, Vec<CatalogueEntry>)] {
        &self.providers
    }

    /// The first entry named `model_id` across all providers, parsed as
    /// usual; a bare `{name, display_name: name}` if there's no such entry
    /// (`_catalogue_entry_for_model` in Python).
    pub fn entry_for_model(&self, model_id: &str) -> CatalogueEntry {
        for (_, entries) in &self.providers {
            if let Some(entry) = entries.iter().find(|e| e.name == model_id) {
                return entry.clone();
            }
        }
        CatalogueEntry { name: model_id.to_string(), display_name: model_id.to_string(), ..Default::default() }
    }
}

/// A plain string is `{name, display_name: name}` with empty kwargs; a
/// mapping needs `name` and takes the rest from its optional fields
/// (`_parse_catalogue_entry` in Python).
fn parse_entry(raw: &serde_yaml_ng::Value) -> Result<CatalogueEntry, String> {
    if let Some(name) = raw.as_str() {
        return Ok(CatalogueEntry { name: name.to_string(), display_name: name.to_string(), ..Default::default() });
    }
    let name = raw
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or("catalogue entry must be a string or a mapping with a 'name'")?
        .to_string();
    let display_name = raw.get("display_name").and_then(|d| d.as_str()).map(str::to_string).unwrap_or_else(|| name.clone());
    let kwargs = |key: &str| -> Result<serde_json::Map<String, serde_json::Value>, String> {
        match raw.get(key) {
            None => Ok(Default::default()),
            Some(v) if v.is_null() => Ok(Default::default()),
            Some(v) => serde_json::to_value(v)
                .map_err(|e| e.to_string())?
                .as_object()
                .cloned()
                .ok_or_else(|| format!("catalogue entry '{key}' must be a mapping")),
        }
    };
    let response_re = raw.get("response_re").and_then(|r| r.as_str()).map(str::to_string);
    let base_url = raw.get("base_url").and_then(|b| b.as_str()).map(str::to_string);
    Ok(CatalogueEntry {
        name,
        display_name,
        processor_kwargs: kwargs("processor_kwargs")?,
        model_kwargs: kwargs("model_kwargs")?,
        generation_kwargs: kwargs("generation_kwargs")?,
        response_re,
        base_url,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
             openai:\n  - gpt-a\n\
             local:\n  - local-a\n",
        )
        .unwrap();
        let names: Vec<&str> = catalogue.providers().iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, vec!["gemini", "openai", "local"]);
        assert_eq!(catalogue.providers()[0].1.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), vec!["gemini-a", "gemini-b"]);
    }

    #[test]
    fn object_entries_take_kwargs_response_re_and_base_url() {
        let yaml = "local:\n\
            \x20 - name: meta-models/Muse-Glimmer-30B\n\
            \x20   generation_kwargs:\n\
            \x20     temperature: 1.0\n\
            \x20     top_k: 64\n\
            \x20   response_re: '<\\|start\\|>assistant to=user<\\|message\\|>(.*?)$'\n";
        let catalogue = Catalogue::parse(yaml).unwrap();
        let entry = &catalogue.providers()[0].1[0];
        assert_eq!(entry.name, "meta-models/Muse-Glimmer-30B");
        assert_eq!(entry.display_name, "meta-models/Muse-Glimmer-30B");
        assert_eq!(entry.generation_kwargs.get("temperature"), Some(&serde_json::json!(1.0)));
        assert_eq!(entry.generation_kwargs.get("top_k"), Some(&serde_json::json!(64)));
        assert!(entry.processor_kwargs.is_empty());
        assert_eq!(entry.response_re.as_deref(), Some("<\\|start\\|>assistant to=user<\\|message\\|>(.*?)$"));
        assert_eq!(entry.base_url, None);
    }

    #[test]
    fn explicit_null_kwargs_become_empty_maps() {
        let yaml = "local:\n\
            \x20 - name: some/model\n\
            \x20   model_kwargs:\n\
            \x20     quantization_config: null\n";
        let catalogue = Catalogue::parse(yaml).unwrap();
        let entry = &catalogue.providers()[0].1[0];
        assert_eq!(entry.model_kwargs.get("quantization_config"), Some(&serde_json::Value::Null));
    }

    #[test]
    fn entry_for_model_finds_across_providers_and_falls_back_to_a_bare_entry() {
        let catalogue = Catalogue::parse(
            "gemini:\n  - gemini-a\n\
             local:\n  - name: org/model\n    display_name: Model\n",
        )
        .unwrap();
        assert_eq!(catalogue.entry_for_model("org/model").display_name, "Model");
        let fallback = catalogue.entry_for_model("unknown/model");
        assert_eq!(fallback.name, "unknown/model");
        assert_eq!(fallback.display_name, "unknown/model");
        assert!(fallback.processor_kwargs.is_empty());
    }

    #[test]
    fn loads_the_real_vlm_models_yaml_without_errors() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("backend/live_photo_commentary/vlm_models.yaml");
        let catalogue = Catalogue::load(&path).unwrap();
        let names: Vec<&str> = catalogue.providers().iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, vec!["gemini", "openai", "local"]);
        let muse = catalogue.entry_for_model("meta-models/Muse-Glimmer-30B");
        assert!(muse.response_re.is_some());
    }
}
