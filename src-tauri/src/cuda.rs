//! NVIDIA GPU detection and installation of a CUDA-enabled PyTorch into an
//! isolated venv under AppData (`backend-cuda/`), never touching the source
//! tree's `pyproject.toml` / `uv.lock`.

#[cfg(not(target_os = "macos"))]
use std::process::{Command, Stdio};
use tauri::{Emitter, Manager};

use crate::antivirus::retry_while_av_blocked;
use crate::flash_attn;
use crate::util::log_to_file;
use crate::uv::uv_command;

pub(crate) struct InstallState {
    pub(crate) uv_path: std::path::PathBuf,
    pub(crate) backend_dir: std::path::PathBuf,
}

/// Isolated directory for the CUDA venv — always in AppData, never the source
/// tree, so `cargo tauri dev` never dirties pyproject.toml or uv.lock.
pub(crate) fn get_cuda_dir(app: &tauri::AppHandle) -> std::path::PathBuf {
    app.path()
        .app_data_dir()
        .expect("app data dir unavailable")
        .join("backend-cuda")
}

fn python_in_venv(venv: &std::path::Path) -> std::path::PathBuf {
    if cfg!(target_os = "windows") {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

/// Returns the Python executable to use for uvicorn:
/// the CUDA venv if installed, otherwise the regular venv.
pub(crate) fn get_python_exe(
    cuda_dir: &std::path::Path,
    backend_dir: &std::path::Path,
) -> std::path::PathBuf {
    let cuda_python = python_in_venv(&cuda_dir.join(".venv"));
    if cuda_dir.join(".cuda_torch").exists() && cuda_python.exists() {
        return cuda_python;
    }
    python_in_venv(&backend_dir.join(".venv"))
}

// ── GPU / driver detection ───────────────────────────────────────────────────

#[cfg(not(target_os = "macos"))]
fn pytorch_cuda_index(major: u32, minor: u32) -> Option<&'static str> {
    match major {
        13.. => Some("cu128"), // CUDA 13.x driver supports cu128 wheels (backward-compatible)
        12 if minor >= 8 => Some("cu128"),
        12 if minor >= 6 => Some("cu126"),
        12 if minor >= 4 => Some("cu124"),
        12 if minor >= 1 => Some("cu121"),
        11 if minor >= 8 => Some("cu118"),
        _ => None,
    }
}

/// Runs `nvidia-smi`, extracts the maximum supported CUDA version, and returns
/// the matching PyTorch wheel index (e.g. `"cu124"`), or `None` if no NVIDIA
/// GPU is detected or the driver is too old.
#[cfg(not(target_os = "macos"))]
fn detect_cuda_index() -> Option<&'static str> {
    let mut cmd = Command::new("nvidia-smi");
    crate::util::no_window(&mut cmd);
    let output = cmd.stdout(Stdio::piped()).stderr(Stdio::null()).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        // Windows nvidia-smi may report "CUDA UMD Version:" instead of "CUDA Version:"
        let key = if line.contains("CUDA Version:") {
            "CUDA Version:"
        } else if line.contains("CUDA UMD Version:") {
            "CUDA UMD Version:"
        } else {
            continue;
        };
        let pos = line.find(key).unwrap();
        let rest = line[pos + key.len()..].trim();
        let version = rest.split_whitespace().next()?;
        let mut parts = version.splitn(2, '.');
        let major: u32 = parts.next()?.parse().ok()?;
        // Strip trailing non-digit chars (pipes, spaces) that appear in the smi table
        let minor_str = parts.next()?;
        let minor: u32 = minor_str
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .ok()?;
        return pytorch_cuda_index(major, minor);
    }
    None
}

/// Detects an NVIDIA GPU and returns the PyTorch CUDA wheel index to suggest to
/// the frontend, or `None` if there's no GPU or the matching CUDA torch is
/// already installed.
#[cfg(not(target_os = "macos"))]
pub(crate) fn detect_cuda_suggestion(handle: &tauri::AppHandle) -> Option<String> {
    let marker = get_cuda_dir(handle).join(".cuda_torch");
    let installed = std::fs::read_to_string(&marker).unwrap_or_default();
    detect_cuda_index().and_then(|cu_index| {
        if installed.trim() != cu_index {
            log::info!("NVIDIA GPU detected, suggesting {cu_index} torch");
            Some(cu_index.to_string())
        } else {
            None
        }
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn detect_cuda_suggestion(_handle: &tauri::AppHandle) -> Option<String> {
    None
}

// ── CUDA torch installation ──────────────────────────────────────────────────

/// Appends (or replaces) the `[tool.uv.sources]` / `[[tool.uv.index]]` blocks
/// in pyproject.toml so that `uv sync` pulls the CUDA-enabled torch wheel.
/// If `flash` is Some, also pins the torch minor version and adds flash-attn as a
/// direct-URL dependency so the pre-built wheel is installed alongside torch.
fn update_pyproject_for_cuda(
    backend_dir: &std::path::Path,
    cu_index: &str,
    flash: Option<&flash_attn::FlashAttnWheel>,
) -> std::io::Result<()> {
    let path = backend_dir.join("pyproject.toml");
    let mut content = std::fs::read_to_string(&path)?;
    // Normalize to LF so the replacement patterns below match regardless of
    // whether the source file was written with CRLF (common on Windows).
    content = content.replace("\r\n", "\n");

    // Inject xformers into [project] dependencies so uv resolves it.
    // Kept out of the source pyproject.toml to avoid failing on platforms without
    // CUDA wheels (Mac/CPU Linux). The [tool.uv.sources] block below points uv at
    // the pytorch index so the CUDA-matched wheel is picked up.
    if !content.contains("\"xformers\"") {
        content = content.replace(
            "  \"torchvision\",\n",
            "  \"torchvision\",\n  \"xformers\",\n",
        );
    }

    // When we have a flash-attn wheel, pin torch to its required minor version so uv
    // doesn't pull a newer torch that flash-attn wasn't built against.
    if let Some(fw) = flash {
        let tv = &fw.torch_version; // e.g. "2.7"
        if let Some((maj, min_str)) = tv.split_once('.') {
            if let Ok(min) = min_str.parse::<u32>() {
                let pinned = format!("torch>={tv},<{maj}.{}", min + 1);
                content = content.replace("  \"torch\",\n", &format!("  \"{pinned}\",\n"));
            }
        }
        if !content.contains("\"flash-attn\"") {
            content = content.replace(
                "  \"xformers\",\n",
                "  \"xformers\",\n  \"flash-attn\",\n",
            );
        }
    }

    // Preserve non-torch/torchvision/xformers/flash-attn entries from any existing
    // [tool.uv.sources] block (e.g. the pyopenjtalk vendor-path entry added by the
    // build scripts).
    let preserved: Vec<String> = if let Some(pos) = content.find("\n[tool.uv.sources]") {
        let tail = &content[pos + 1..]; // skip the leading '\n'
        let body_start = tail.find('\n').map(|p| p + 1).unwrap_or(tail.len());
        let body = &tail[body_start..];
        // Section ends at the next TOML table header ('[' at start of a line).
        let body_end = body.find("\n[").map(|p| p + 1).unwrap_or(body.len());
        body[..body_end]
            .lines()
            .filter(|l| {
                let t = l.trim();
                !t.is_empty()
                    && !t.starts_with("torch ")
                    && !t.starts_with("torchvision ")
                    && !t.starts_with("xformers ")
                    && !t.starts_with("flash-attn ")
            })
            // vendor/ is relative to backend/; rewrite for backend-cuda/ (its sibling).
            .map(|l| l.replace("\"vendor/", "\"../backend/vendor/"))
            .collect()
    } else {
        vec![]
    };

    // Strip the old block so re-runs are idempotent.
    if let Some(pos) = content.find("\n[tool.uv.sources]") {
        content.truncate(pos);
    }

    let idx = format!("pytorch-{cu_index}");
    let platform = "sys_platform == 'win32' or sys_platform == 'linux'";
    content.push_str(&format!(
        "\n[tool.uv.sources]\ntorch       = [{{ index = \"{idx}\", marker = \"{platform}\" }}]\ntorchvision = [{{ index = \"{idx}\", marker = \"{platform}\" }}]\n"
    ));
    if let Some(fw) = flash {
        content.push_str(&format!("flash-attn  = {{ url = \"{}\" }}\n", fw.url));
    }
    for line in &preserved {
        content.push_str(line);
        content.push('\n');
    }
    content.push_str(&format!(
        "\n[[tool.uv.index]]\nname = \"{idx}\"\nurl  = \"https://download.pytorch.org/whl/{cu_index}\"\nexplicit = true\n"
    ));

    std::fs::write(path, content)
}

#[tauri::command]
pub async fn install_cuda_torch(
    cu_index: String,
    app: tauri::AppHandle,
    state: tauri::State<'_, InstallState>,
) -> Result<(), String> {
    let uv = state.uv_path.clone();
    let source_backend = state.backend_dir.clone();
    let cuda_dir = get_cuda_dir(&app);
    let marker = cuda_dir.join(".cuda_torch");

    // Create an isolated directory so we never modify the source-tree files.
    std::fs::create_dir_all(&cuda_dir)
        .map_err(|e| format!("Failed to create CUDA dir: {e}"))?;

    // Copy pyproject.toml from source (always fresh so the base is clean).
    std::fs::copy(
        source_backend.join("pyproject.toml"),
        cuda_dir.join("pyproject.toml"),
    ).map_err(|e| format!("Failed to copy pyproject.toml: {e}"))?;

    // Seed the resolver with the existing lock file to speed up resolution.
    if source_backend.join("uv.lock").exists() {
        let _ = std::fs::copy(source_backend.join("uv.lock"), cuda_dir.join("uv.lock"));
    }

    let _ = app.emit("cuda_install_progress", "Checking for flash-attn pre-built wheels…");
    // Python version is fixed to 3.12 by requires-python in pyproject.toml.
    let flash_wheel = flash_attn::find_flash_attn_wheel(&cu_index, "312").await;
    match &flash_wheel {
        Some(fw) => log::info!(
            "flash_attn: found {} for torch {} — will install alongside CUDA torch",
            fw.flash_version, fw.torch_version
        ),
        None => log::info!("flash_attn: no matching wheel for {cu_index}; skipping"),
    }

    update_pyproject_for_cuda(&cuda_dir, &cu_index, flash_wheel.as_ref())
        .map_err(|e| format!("Failed to update pyproject.toml: {e}"))?;

    let _ = app.emit("cuda_install_progress", "Downloading CUDA PyTorch — this may take a few minutes…");

    let cu = cu_index.clone();
    let has_flash = flash_wheel.is_some();
    let status = tokio::task::spawn_blocking(move || {
        let mut cmd = uv_command(&uv);
        let mut sync_args = vec![
            "sync",
            "--no-install-project", // source code loaded via PYTHONPATH, not editable install
            "--reinstall-package", "torch",
            "--reinstall-package", "torchvision",
            "--reinstall-package", "xformers",
        ];
        if has_flash {
            sync_args.extend_from_slice(&["--reinstall-package", "flash-attn"]);
        }
        cmd.args(&sync_args)
        .current_dir(&cuda_dir);
        let (out, err) = log_to_file(&cuda_dir.join("uv-sync.log"));
        cmd.stdout(out).stderr(err);
        crate::util::no_window(&mut cmd);
        // uv itself, or the DLLs it just extracted, may still be getting
        // scanned by Defender right after a previous reinstall — retry
        // briefly rather than failing outright.
        retry_while_av_blocked(|| cmd.status())
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    if status.success() {
        let _ = std::fs::write(&marker, &cu);
        let _ = app.emit("cuda_install_done", &cu);
        Ok(())
    } else {
        let msg = format!("uv sync failed (exit code {:?})", status.code());
        let _ = app.emit("cuda_install_failed", &msg);
        Err(msg)
    }
}
