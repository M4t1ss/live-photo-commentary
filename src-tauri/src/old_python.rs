//! The Python environment of versions before 0.2.0, which the app offers to
//! delete once (it is several GB: a virtual environment with PyTorch).
//!
//! Python-era installs kept the backend in `<app data>/backend` (source, `.venv`,
//! `.env`) and, after installing CUDA PyTorch, an extra environment in
//! `<app data>/backend-cuda`. Settings, promptsets and uploaded avatar models
//! were already moved out of `backend` when this version first started
//! (`config`, `promptsets` and `avatar` migrations).
//!
//! uv's managed Python (`%LOCALAPPDATA%\uv\python` on Windows) is not touched:
//! it may be in use by other projects.

use std::path::{Path, PathBuf};

/// Created when the user answers "Keep"; the offer is then never made again.
const DECLINED_MARKER: &str = "old-python-env-kept";

/// What stays in `backend` when it is cleaned up: the migrations only move
/// these folders if the new location did not exist yet, so a folder that is
/// still here may hold files that were never migrated.
const KEEP_IN_BACKEND: [&str; 2] = ["prompts", "user_models"];

fn backend_dir(app_data_dir: &Path) -> PathBuf {
    crate::state::old_python_backend_dir(app_data_dir)
}

fn cuda_dir(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("backend-cuda")
}

/// The old environment's folders that exist. `backend` only counts if it looks
/// like a Python backend, so a folder that happens to share the name is never
/// offered for deletion.
fn folders(app_data_dir: &Path) -> Vec<PathBuf> {
    let backend = backend_dir(app_data_dir);
    let cuda = cuda_dir(app_data_dir);
    let mut found = Vec::new();
    if backend.join(".venv").is_dir() || backend.join("pyproject.toml").is_file() {
        found.push(backend);
    }
    if cuda.join(".venv").is_dir() || cuda.join(".cuda_torch").exists() {
        found.push(cuda);
    }
    found
}

/// The size in bytes of the old environment, if there is one and the user has
/// not said to keep it. `settings_path` must exist: until then the migration
/// has not run and the old folders still hold the only copy of the settings.
pub fn found(app_data_dir: &Path, settings_path: &Path) -> Option<u64> {
    if !settings_path.is_file() || app_data_dir.join(DECLINED_MARKER).exists() {
        return None;
    }
    let folders = folders(app_data_dir);
    if folders.is_empty() {
        return None;
    }
    Some(folders.iter().map(|folder| size(folder)).sum())
}

/// Deletes the old environment.
pub fn remove(app_data_dir: &Path) -> Result<(), String> {
    let mut errors = Vec::new();
    for folder in folders(app_data_dir) {
        let result = if folder == backend_dir(app_data_dir) { clean_backend(&folder) } else { std::fs::remove_dir_all(&folder) };
        if let Err(error) = result {
            // Typically a file in use: the old backend can still be running.
            errors.push(format!("{}: {error}", folder.display()));
        }
    }
    if errors.is_empty() { Ok(()) } else { Err(format!("could not delete everything: {}", errors.join("; "))) }
}

/// Deletes everything in `backend` except `KEEP_IN_BACKEND`, then the folder
/// itself if nothing is left.
fn clean_backend(backend: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(backend)? {
        let entry = entry?;
        if KEEP_IN_BACKEND.iter().any(|keep| entry.file_name() == *keep) {
            continue;
        }
        if entry.file_type()?.is_dir() {
            std::fs::remove_dir_all(entry.path())?;
        } else {
            std::fs::remove_file(entry.path())?;
        }
    }
    // Fails if something was kept, which is fine.
    let _ = std::fs::remove_dir(backend);
    Ok(())
}

/// Records that the user wants to keep the old environment.
pub fn keep(app_data_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(app_data_dir).map_err(|e| e.to_string())?;
    std::fs::write(app_data_dir.join(DECLINED_MARKER), "").map_err(|e| e.to_string())
}

/// Total size of the files below `folder`, not following links.
fn size(folder: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(folder) else { return 0 };
    entries
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => size(&entry.path()),
            Ok(kind) if kind.is_file() => entry.metadata().map(|m| m.len()).unwrap_or(0),
            _ => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An app data folder with a Python-era install in it.
    fn old_install(name: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("lpc-old-python-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let backend = dir.join("backend");
        std::fs::create_dir_all(backend.join(".venv/Lib")).unwrap();
        std::fs::write(backend.join(".venv/Lib/torch.dll"), vec![0u8; 1000]).unwrap();
        std::fs::write(backend.join("pyproject.toml"), "x").unwrap();
        std::fs::write(backend.join(".env"), "VLM_PROVIDER=local").unwrap();
        std::fs::create_dir_all(dir.join("backend-cuda/.venv")).unwrap();
        std::fs::write(dir.join("backend-cuda/.cuda_torch"), "").unwrap();
        let settings = dir.join("settings.json");
        std::fs::write(&settings, "{}").unwrap();
        (dir, settings)
    }

    #[test]
    fn finds_the_old_environment_and_adds_up_its_size() {
        let (dir, settings) = old_install("found");
        assert_eq!(found(&dir, &settings), Some(1000 + 1 + 18));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nothing_is_found_before_the_settings_were_migrated() {
        let (dir, settings) = old_install("unmigrated");
        std::fs::remove_file(&settings).unwrap();
        assert_eq!(found(&dir, &settings), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_backend_folder_that_is_not_python_is_left_alone() {
        let dir = std::env::temp_dir().join(format!("lpc-old-python-other-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(dir.join("backend")).unwrap();
        std::fs::write(dir.join("backend/data.txt"), "mine").unwrap();
        let settings = dir.join("settings.json");
        std::fs::write(&settings, "{}").unwrap();
        assert_eq!(found(&dir, &settings), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn keeping_it_ends_the_offer() {
        let (dir, settings) = old_install("keep");
        keep(&dir).unwrap();
        assert_eq!(found(&dir, &settings), None);
        assert!(dir.join("backend/.venv").is_dir(), "keeping deletes nothing");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn removing_it_keeps_folders_that_were_not_migrated() {
        let (dir, _) = old_install("remove");
        std::fs::create_dir_all(dir.join("backend/prompts")).unwrap();
        std::fs::write(dir.join("backend/prompts/mine.yaml"), "x").unwrap();
        remove(&dir).unwrap();
        assert!(!dir.join("backend-cuda").exists());
        assert!(!dir.join("backend/.venv").exists());
        assert!(!dir.join("backend/.env").exists());
        assert!(dir.join("backend/prompts/mine.yaml").is_file());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn removing_it_deletes_the_backend_folder_when_nothing_is_kept() {
        let (dir, settings) = old_install("remove-all");
        remove(&dir).unwrap();
        assert!(!dir.join("backend").exists());
        assert_eq!(found(&dir, &settings), None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
