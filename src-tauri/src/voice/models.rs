//! The speech recognition models the user can choose between: whisper.cpp's
//! multilingual models from Hugging Face, quantized to keep them small and
//! fast on the CPU. Downloaded the first time they are needed.

use std::path::PathBuf;

/// Serialized for the Settings list: only what it shows.
#[derive(serde::Serialize)]
pub struct AsrModel {
    /// What the `voice_model` setting stores.
    pub name: &'static str,
    pub label: &'static str,
    #[serde(skip)]
    pub repo: &'static str,
    #[serde(skip)]
    pub file: &'static str,
}

pub const DEFAULT_MODEL: &str = "base";

pub const MODELS: [AsrModel; 4] = [
    AsrModel { name: "tiny", label: "Tiny (32 MB, fastest, least accurate)", repo: "ggerganov/whisper.cpp", file: "ggml-tiny-q5_1.bin" },
    AsrModel { name: "base", label: "Base (60 MB)", repo: "ggerganov/whisper.cpp", file: "ggml-base-q5_1.bin" },
    AsrModel { name: "small", label: "Small (190 MB, more accurate, slower)", repo: "ggerganov/whisper.cpp", file: "ggml-small-q5_1.bin" },
    AsrModel {
        name: "large-v3-turbo",
        label: "Large v3 turbo (574 MB, most accurate, slowest on the CPU)",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-large-v3-turbo-q5_0.bin",
    },
];

/// The model called `name`, or the default one for a name that is not (or no
/// longer) in the list.
pub fn find(name: &str) -> &'static AsrModel {
    MODELS.iter().find(|m| m.name == name).unwrap_or_else(|| {
        MODELS.iter().find(|m| m.name == DEFAULT_MODEL).expect("the default model should be in the list")
    })
}

/// The model file's local path, downloading it first if it isn't cached.
/// `progress` gets the file name, the bytes downloaded and the file's size.
pub async fn download(model: &AsrModel, progress: vlm_describer::DownloadProgress) -> Result<PathBuf, String> {
    vlm_describer::download_repo_file(model.repo, model.file, Some(progress)).await.map_err(|e| {
        // The cause (no network, no such file) is in the error's source.
        match std::error::Error::source(&e) {
            Some(cause) => format!("{e}: {cause}"),
            None => e.to_string(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_in_the_list_and_names_are_unique() {
        assert!(MODELS.iter().any(|m| m.name == DEFAULT_MODEL));
        for (i, a) in MODELS.iter().enumerate() {
            assert!(MODELS.iter().skip(i + 1).all(|b| a.name != b.name), "{} appears twice", a.name);
        }
    }

    #[test]
    fn the_list_for_settings_has_names_and_labels_only() {
        let json = serde_json::to_value(MODELS.iter().collect::<Vec<_>>()).unwrap();
        assert_eq!(json[1], serde_json::json!({ "name": "base", "label": "Base (60 MB)" }));
    }

    #[test]
    fn an_unknown_name_gets_the_default() {
        assert_eq!(find("small").file, "ggml-small-q5_1.bin");
        assert_eq!(find("no-such-model").name, DEFAULT_MODEL);
        assert_eq!(find("").name, DEFAULT_MODEL);
    }

    #[test]
    fn every_model_is_a_multilingual_quantized_ggml_file() {
        for m in &MODELS {
            assert!(m.file.starts_with("ggml-") && m.file.ends_with(".bin"), "{}", m.file);
            assert!(!m.file.contains(".en"), "{} is English only; the language setting needs multilingual", m.file);
            assert!(m.file.contains("-q5_"), "{} is not quantized", m.file);
        }
    }
}
