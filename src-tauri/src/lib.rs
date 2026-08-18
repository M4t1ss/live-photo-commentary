use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager, State};
use tauri_plugin_window_state::WindowExt;

// ── App state ────────────────────────────────────────────────────────────────

struct BackendState {
    port: u16,
    process: BackendProcess,
    cuda_suggestion: Option<String>,
}

/// Wraps the child process handle so we can share it between the shutdown handler
/// (which kills on normal exit), the Drop impl (safety net), and the monitor
/// thread (which detects crashes).
struct BackendProcess {
    inner: Arc<Mutex<Option<Child>>>,
}

impl BackendProcess {
    /// Kills the backend process and its children.
    /// Safe to call multiple times — the Option is taken on first call;
    /// subsequent calls find None and return immediately.
    fn kill(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(mut child) = guard.take() {
                #[cfg(target_os = "windows")]
                {
                    let mut cmd = Command::new("taskkill");
                    no_window(cmd.args(["/F", "/T", "/PID", &child.id().to_string()]));
                    let _ = cmd.status();
                }
                // Kill the whole process group so uvicorn (and any model it loaded)
                // are also terminated — not just the uv launcher. The child was
                // started with process_group(0), so its pgid equals its pid.
                #[cfg(unix)]
                {
                    let pgid = child.id();
                    let _ = Command::new("kill")
                        .args(["-KILL", &format!("-{pgid}")])
                        .status();
                }
                let _ = child.wait();
            }
            // None → already killed (by RunEvent::Exit handler or monitor thread)
        }
    }
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        self.kill(); // safety net — primary kill happens in RunEvent::Exit
    }
}

// ── Tauri commands ────────────────────────────────────────────────────────────

#[tauri::command]
fn get_backend_port(state: State<BackendState>) -> (u16, Option<String>) {
    (state.port, state.cuda_suggestion.clone())
}

#[tauri::command]
fn restart_backend(
    app: tauri::AppHandle,
    state: State<'_, BackendState>,
    install: State<'_, InstallState>,
) {
    let port = state.port;
    let inner = Arc::clone(&state.process.inner);
    let uv = install.uv_path.clone();
    state.process.kill();
    tauri::async_runtime::spawn(async move {
        // Re-run setup (which retries `uv sync` if `.venv` is missing or
        // incomplete) rather than assuming the previous install is intact —
        // a crashed backend often means the venv itself is broken.
        if let Some(backend_dir) = resolve_backend_dir(&app, &uv).await {
            spawn_and_monitor_backend(app, port, uv, backend_dir, inner).await;
        }
    });
}

// ── Screenshot ───────────────────────────────────────────────────────────────

fn png_to_data_url(path: &std::path::Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read failed: {e}"))?;
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:image/png;base64,{b64}"))
}

#[tauri::command]
fn list_monitors() -> Result<Vec<MonitorInfo>, String> {
    use screenshots::Screen;
    let screens = Screen::all().map_err(|e| format!("screen enumeration failed: {e}"))?;
    Ok(screens
        .into_iter()
        .enumerate()
        .map(|(i, s)| MonitorInfo {
            index: i,
            name: format!("Display {}", s.display_info.id),
            x: s.display_info.x,
            y: s.display_info.y,
            width: s.display_info.width,
            height: s.display_info.height,
            is_primary: s.display_info.is_primary,
        })
        .collect())
}

#[tauri::command]
fn capture_monitor_preview(
    app: tauri::AppHandle,
    state: State<'_, ScreenshotState>,
    monitor_idx: Option<u32>,
) -> Result<String, String> {
    let preview_path = state.frames_dir.join("frame_preview.png");
    if state.is_wsl {
        let win_dest = wslpath_to_windows(&preview_path)?;
        let exe = locate_screenshot_exe(&app);
        let mut cmd = Command::new(&exe);
        cmd.arg(&win_dest);
        if let Some(idx) = monitor_idx {
            cmd.args(["--monitor", &idx.to_string()]);
        }
        let status = cmd.status().map_err(|e| format!("screenshot.exe failed: {e}"))?;
        if !status.success() {
            return Err(format!("screenshot.exe exited with {:?}", status.code()));
        }
    } else {
        use screenshots::Screen;
        let screens = Screen::all().map_err(|e| format!("screen enumeration failed: {e}"))?;
        let idx = monitor_idx.unwrap_or(0) as usize;
        let screen = screens.get(idx).or_else(|| screens.first()).ok_or("no screens found")?;
        screen
            .capture()
            .map_err(|e| format!("capture failed: {e}"))?
            .save(&preview_path)
            .map_err(|e| format!("save failed: {e}"))?;
    }
    png_to_data_url(&preview_path)
}

#[tauri::command]
fn take_screenshot(
    app: tauri::AppHandle,
    state: State<'_, ScreenshotState>,
    monitor_idx: Option<u32>,
    clip: Option<[i32; 4]>,
) -> Result<String, String> {
    let slot = state.frame_slot.fetch_xor(1, Ordering::Relaxed);
    let dest = state.frames_dir.join(format!("frame_{slot}.png"));

    let result = (|| -> Result<String, String> {
        if state.is_wsl {
            let exe = locate_screenshot_exe(&app);
            let win_dest = wslpath_to_windows(&dest)?;
            let mut cmd = Command::new(&exe);
            cmd.arg(&win_dest);
            if let Some(idx) = monitor_idx {
                cmd.args(["--monitor", &idx.to_string()]);
            }
            if let Some([x, y, w, h]) = clip {
                cmd.args(["--clip", &format!("{x},{y},{w},{h}")]);
            }
            let status = cmd
                .status()
                .map_err(|e| format!("screenshot.exe failed to start: {e}"))?;
            if !status.success() {
                return Err(format!("screenshot.exe exited with {:?}", status.code()));
            }
        } else {
            use screenshots::Screen;
            let screens = Screen::all().map_err(|e| format!("screen enumeration failed: {e}"))?;
            let idx = monitor_idx.unwrap_or(0) as usize;
            let screen = screens.get(idx).or_else(|| screens.first()).ok_or("no screens found")?;
            let img = screen.capture().map_err(|e| format!("capture failed: {e}"))?;
            let img = if let Some([x, y, w, h]) = clip {
                let ix = x.max(0) as u32;
                let iy = y.max(0) as u32;
                let iw = (w as u32).min(img.width().saturating_sub(ix)).max(1);
                let ih = (h as u32).min(img.height().saturating_sub(iy)).max(1);
                image::DynamicImage::ImageRgba8(img).crop_imm(ix, iy, iw, ih)
                    .into_rgba8()
            } else {
                img
            };
            img.save(&dest).map_err(|e| format!("save failed: {e}"))?;
        }
        Ok(dest.to_string_lossy().into_owned())
    })();

    if let Err(msg) = &result {
        log::error!("take_screenshot (slot {slot}) failed: {msg}");
    }
    result
}

// ── Port discovery ────────────────────────────────────────────────────────────

fn find_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

// ── CUDA detection (Windows / Linux only) ────────────────────────────────────

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
    no_window(&mut cmd);
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

// ── CUDA torch installation ───────────────────────────────────────────────────

/// Appends (or replaces) the `[tool.uv.sources]` / `[[tool.uv.index]]` blocks
/// in pyproject.toml so that `uv sync` pulls the CUDA-enabled torch wheel.
fn update_pyproject_for_cuda(backend_dir: &std::path::Path, cu_index: &str) -> std::io::Result<()> {
    let path = backend_dir.join("pyproject.toml");
    let mut content = std::fs::read_to_string(&path)?;

    // Preserve non-torch entries from any existing [tool.uv.sources] block
    // (e.g. the pyopenjtalk vendor-path entry added by the build scripts).
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
                !t.is_empty() && !t.starts_with("torch ") && !t.starts_with("torchvision ")
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
    for line in &preserved {
        content.push_str(line);
        content.push('\n');
    }
    content.push_str(&format!(
        "\n[[tool.uv.index]]\nname = \"{idx}\"\nurl  = \"https://download.pytorch.org/whl/{cu_index}\"\nexplicit = true\n"
    ));

    std::fs::write(path, content)
}

struct InstallState {
    uv_path: std::path::PathBuf,
    backend_dir: std::path::PathBuf,
}

struct ScreenshotState {
    frames_dir: std::path::PathBuf,
    frame_slot: AtomicUsize,
    is_wsl: bool,
}

#[derive(serde::Serialize)]
struct MonitorInfo {
    index: usize,
    name: String,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    is_primary: bool,
}

fn detect_wsl() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|v| v.to_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

fn locate_screenshot_exe(app: &tauri::AppHandle) -> std::path::PathBuf {
    if cfg!(debug_assertions) {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("CARGO_MANIFEST_DIR has no parent")
            .join("screenshot.exe")
    } else {
        app.path()
            .resource_dir()
            .expect("resource dir unavailable")
            .join("resources")
            .join("screenshot.exe")
    }
}

fn wslpath_to_windows(path: &std::path::Path) -> Result<String, String> {
    let out = Command::new("wslpath")
        .arg("-w")
        .arg(path)
        .output()
        .map_err(|e| format!("wslpath failed: {e}"))?;
    if !out.status.success() {
        return Err(format!("wslpath error: {}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Isolated directory for the CUDA venv — always in AppData, never the source
/// tree, so `cargo tauri dev` never dirties pyproject.toml or uv.lock.
fn get_cuda_dir(app: &tauri::AppHandle) -> std::path::PathBuf {
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
fn get_python_exe(
    cuda_dir: &std::path::Path,
    backend_dir: &std::path::Path,
) -> std::path::PathBuf {
    let cuda_python = python_in_venv(&cuda_dir.join(".venv"));
    if cuda_dir.join(".cuda_torch").exists() && cuda_python.exists() {
        return cuda_python;
    }
    python_in_venv(&backend_dir.join(".venv"))
}

#[tauri::command]
async fn install_cuda_torch(
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

    update_pyproject_for_cuda(&cuda_dir, &cu_index)
        .map_err(|e| format!("Failed to update pyproject.toml: {e}"))?;

    let _ = app.emit("cuda_install_progress", "Downloading CUDA PyTorch — this may take a few minutes…");

    let cu = cu_index.clone();
    let status = tokio::task::spawn_blocking(move || {
        let mut cmd = uv_command(&uv);
        cmd.args([
            "sync",
            "--no-install-project", // source code loaded via PYTHONPATH, not editable install
            "--reinstall-package", "torch",
            "--reinstall-package", "torchvision",
        ])
        .current_dir(&cuda_dir);
        let (out, err) = log_to_file(&cuda_dir.join("uv-sync.log"));
        cmd.stdout(out).stderr(err);
        no_window(&mut cmd);
        cmd.status()
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

// ── uv helpers ───────────────────────────────────────────────────────────────

fn uv_command(uv: &std::path::Path) -> Command {
    // On Windows, AppData\Roaming is often redirected by OneDrive's Known Folder
    // Move, which creates reparse points that Windows refuses to traverse (error
    // 448). uv defaults to installing managed Pythons under AppData\Roaming\uv,
    // so redirect it to AppData\Local\uv which OneDrive does not touch.
    #[cfg(target_os = "windows")]
    {
        let mut cmd = Command::new(uv);
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            let python_dir = std::path::Path::new(&local).join("uv").join("python");
            cmd.env("UV_PYTHON_INSTALL_DIR", python_dir);
        }
        cmd
    }

    #[cfg(not(target_os = "windows"))]
    {
        Command::new(uv)
    }
}

// ── Process helpers ───────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Suppresses the console window for a spawned child process on Windows.
/// No-op on other platforms.
fn no_window(_cmd: &mut Command) {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        _cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

/// Opens `path` as a combined stdout+stderr log file.
/// Falls back to null handles if the file cannot be created.
fn log_to_file(path: &std::path::Path) -> (Stdio, Stdio) {
    std::fs::File::create(path)
        .ok()
        .and_then(|f| f.try_clone().ok().map(|c| (Stdio::from(f), Stdio::from(c))))
        .unwrap_or_else(|| (Stdio::null(), Stdio::null()))
}

// ── Antivirus detection helpers ───────────────────────────────────────────────

/// Returns a hint string when an IO error looks like Windows Defender or
/// antivirus software blocked an executable from running.
fn av_blocked_hint(e: &std::io::Error) -> Option<&'static str> {
    #[cfg(target_os = "windows")]
    // 5   = ERROR_ACCESS_DENIED  (Defender blocked execution of a quarantined binary)
    // 225 = ERROR_VIRUS_INFECTED (file flagged as a threat)
    // 226 = ERROR_VIRUS_DELETED  (file partially removed by Defender)
    if matches!(e.raw_os_error(), Some(5) | Some(225) | Some(226)) {
        return Some(
            "Windows Defender or antivirus may have blocked the executable. \
             Open Windows Security → Virus & threat protection → Protection history \
             to check, or add the app data folder to Defender exclusions and restart.",
        );
    }
    let _ = e;
    None
}

/// Renames broken managed Python installations so uv skips them and downloads
/// a clean copy on the next sync attempt.
///
/// A healthy install has `Lib/` (install_only format) or a `python3*.zip`
/// alongside `python.exe`.  If neither exists the extraction was interrupted
/// (e.g. Defender scanned files mid-write) and the install is unusable.
///
/// `std::fs::rename` succeeds even when DLL files inside the directory are
/// locked by a running process, unlike `remove_dir_all` which returns
/// ERROR_ACCESS_DENIED on locked files.  Renaming the directory makes uv
/// treat it as absent and triggers a fresh download on the next launch.
#[cfg(target_os = "windows")]
fn quarantine_broken_python_installs(python_dir: &std::path::Path) {
    let entries = match std::fs::read_dir(python_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !path.is_dir()
            || !name_str.starts_with("cpython-")
            || name_str.contains("-broken")
        {
            continue;
        }
        if !path.join("python.exe").exists() {
            continue;
        }
        let has_lib = path.join("Lib").exists();
        let has_stdlib_zip = std::fs::read_dir(&path)
            .ok()
            .map(|d| {
                d.flatten().any(|e| {
                    let n = e.file_name();
                    let s = n.to_string_lossy();
                    s.starts_with("python3") && s.ends_with(".zip")
                })
            })
            .unwrap_or(false);
        if !has_lib && !has_stdlib_zip {
            let broken_name = format!("{}-broken", name_str);
            let broken = python_dir.join(&broken_name);
            log::warn!(
                "Python install at {} is missing stdlib (no Lib/ or python3*.zip) — \
                 renaming to {} so uv downloads a clean copy on the next launch.",
                path.display(),
                broken_name,
            );
            if let Err(e) = std::fs::rename(&path, &broken) {
                log::error!("Failed to quarantine broken Python install: {e}");
            }
        }
    }
}

fn quarantine_local_python() {
    #[cfg(target_os = "windows")]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let py_dir = std::path::Path::new(&local).join("uv").join("python");
        quarantine_broken_python_installs(&py_dir);
    }
}

/// Returns a hint string when a process exits very shortly after spawning.
/// An immediate exit is a common pattern when AV software terminates a
/// process as it launches.
fn quick_exit_hint(elapsed: std::time::Duration) -> &'static str {
    #[cfg(target_os = "windows")]
    if elapsed < std::time::Duration::from_secs(3) {
        return " Windows Defender or antivirus may have terminated the process immediately. \
                Open Windows Security → Virus & threat protection → Protection history \
                to check, or add the app data folder to Defender exclusions and restart.";
    }
    let _ = elapsed;
    ""
}

// ── uv binary location ────────────────────────────────────────────────────────

/// Returns the path to the uv binary.
///
/// Production: `tauri.conf.json` bundles `resources/uv-<triple>` via the glob
/// `"resources/uv-*"`. With Tauri's array resource format the file lands at
/// `resource_dir()/resources/uv-<triple>`, so we probe that platform-specific
/// name at runtime.
///
/// Development / fallback: the resources directory won't contain the binary
/// (it's only downloaded by CI), so we fall back to whichever `uv` is on PATH.
fn get_uv_path(app: &tauri::AppHandle) -> std::path::PathBuf {
    // Compile-time selection of the bundled resource path.
    // cfg!() chains are evaluated at compile time; dead arms are erased.
    let resource_rel = if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        "resources/uv-x86_64-pc-windows-msvc.exe"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "resources/uv-x86_64-unknown-linux-gnu"
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "resources/uv-aarch64-apple-darwin"
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        "resources/uv-x86_64-apple-darwin"
    } else {
        "" // unsupported platform — will fall through to system uv
    };

    if !resource_rel.is_empty() {
        if let Ok(resource_dir) = app.path().resource_dir() {
            let bundled = resource_dir.join(resource_rel);
            if bundled.exists() {
                log::info!("Using bundled uv: {}", bundled.display());
                return bundled;
            }
        }
    }

    log::info!("Bundled uv not found — falling back to system uv");
    std::path::PathBuf::from("uv")
}

// ── Backend directory & setup ─────────────────────────────────────────────────

/// Returns the writable backend directory for the current mode:
/// - Debug: the source tree (`backend/` next to `src-tauri/`)
/// - Release: `<app_data_dir>/backend/` — always writable, so `uv sync` can
///   create `.venv` there even on read-only install paths (Program Files, .app)
fn get_backend_dir(app: &tauri::AppHandle) -> std::path::PathBuf {
    if cfg!(debug_assertions) {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("CARGO_MANIFEST_DIR has no parent")
            .join("backend")
    } else {
        app.path()
            .app_data_dir()
            .expect("app data dir unavailable")
            .join("backend")
    }
}

/// Recursively copies `src` into `dst`, creating directories as needed.
fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let dst_path = dst.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir_recursive(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), dst_path)?;
        }
    }
    Ok(())
}

/// Prepares the backend directory and returns its path.
///
/// In release builds:
///   1. Compares the `.app_version` marker in the backend directory against
///      the current app version (from `CARGO_PKG_VERSION`). If missing or
///      different — first launch, upgrade, or interrupted copy — copies the
///      bundled source from `<resource_dir>/resources/backend/` to the
///      writable app data location, then writes the new version marker and
///      removes `pyvenv.cfg` to force a venv rebuild with the new `uv.lock`.
///   2. Runs `uv sync` if `.venv` is absent or incomplete.
///
/// Intended to run on a blocking thread (via `spawn_blocking`) — never call
/// from the main event loop, as `uv sync` can take minutes on first launch.
fn setup_backend(app: &tauri::AppHandle, uv: &std::path::Path) -> Result<std::path::PathBuf, String> {
    const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

    let backend_dir = get_backend_dir(app);

    if !cfg!(debug_assertions) {
        // Re-copy whenever the stored version doesn't match the running app.
        // This handles: first launch (.app_version absent), version upgrades,
        // and interrupted copies (directory created but files missing).
        // AppData persists across reinstalls and uninstalls on Windows, so we
        // cannot rely on directory existence alone.
        let stored_version = std::fs::read_to_string(backend_dir.join(".app_version"))
            .unwrap_or_default();
        let needs_copy = stored_version.trim() != APP_VERSION;

        if needs_copy {
            let resource_backend = app
                .path()
                .resource_dir()
                .expect("resource dir unavailable")
                .join("resources")
                .join("backend");

            if resource_backend.exists() {
                log::info!(
                    "Copying backend source ({APP_VERSION}) to {} …",
                    backend_dir.display()
                );
                let _ = app.emit("setup_progress", "Updating backend…");
                if let Err(e) = copy_dir_recursive(&resource_backend, &backend_dir) {
                    log::error!("Failed to copy backend source: {e}");
                    return Ok(backend_dir);
                }
                // Mark the version so we skip the copy on the next launch.
                let _ = std::fs::write(backend_dir.join(".app_version"), APP_VERSION);
                // uv.lock may have changed — force a venv rebuild.
                let _ = std::fs::remove_file(
                    backend_dir.join(".venv").join("pyvenv.cfg"),
                );
            } else {
                log::error!(
                    "Bundled backend not found at {} — backend will not start",
                    resource_backend.display()
                );
                return Ok(backend_dir);
            }
        }
    }

    // Run `uv sync` if the venv is absent or incomplete.
    // Checking pyvenv.cfg (not just the .venv dir) catches partial installs
    // where the directory was created but uv sync didn't finish.
    if !backend_dir.join(".venv").join("pyvenv.cfg").exists() {
        // On Windows, executables in .venv\Scripts stay locked while any
        // process using them is alive (e.g. a previous session's uvicorn).
        // Remove the stale directory before uv sync so it can start clean;
        // if removal fails the sync attempt below may still succeed or will
        // produce the same error with a useful hint in the log.
        let venv_dir = backend_dir.join(".venv");
        if venv_dir.exists() {
            log::info!("Removing stale .venv before sync…");
            if let Err(e) = std::fs::remove_dir_all(&venv_dir) {
                log::error!(
                    "Cannot remove stale .venv ({e}). \
                     If a Python process from a previous session is still \
                     running, kill it (e.g. `Get-Process python* | Stop-Process -Force`) \
                     and restart the app."
                );
            }
        }

        // Quarantine any broken Python installs before sync so uv downloads a
        // clean copy instead of looping on a corrupt interpreter.  Rename
        // works even when DLL files are held open by a running process.
        quarantine_local_python();

        log::info!("Running `uv sync` in {} …", backend_dir.display());
        let _ = app.emit("setup_progress", "Setting up Python environment…");

        let mut sync_cmd = uv_command(uv);
        sync_cmd.arg("sync");
        sync_cmd.current_dir(&backend_dir);
        // Verbose output so uv-sync.log shows where each package comes from
        // (pre-built wheel path vs. PyPI download vs. source build).
        if !cfg!(debug_assertions) {
            sync_cmd.arg("--verbose");
        }

        // In release: suppress console window on Windows, redirect output to a
        // log file so failures are diagnosable without attaching a debugger.
        if !cfg!(debug_assertions) {
            no_window(&mut sync_cmd);
            let (out, err) = log_to_file(&backend_dir.join("uv-sync.log"));
            sync_cmd.stdout(out).stderr(err);
        }

        match sync_cmd.status() {
            Ok(s) if s.success() => log::info!("Python environment ready."),
            Ok(s) => {
                // Remove the partial venv so the next launch retries rather
                // than silently using an incomplete environment.
                let _ = std::fs::remove_dir_all(backend_dir.join(".venv"));
                // Quarantine the broken Python install (if any) so the next
                // launch downloads a fresh copy.  `remove_dir_all` silently
                // fails when DLL files are locked; rename works in that case.
                quarantine_local_python();
                let msg = format!(
                    "Setup failed: `uv sync` failed (exit code {:?}). See {}",
                    s.code(),
                    backend_dir.join("uv-sync.log").display()
                );
                log::error!("{msg}");
                let _ = app.emit("setup_progress", &msg);
                return Err(msg);
            }
            Err(e) => {
                let hint = av_blocked_hint(&e).map(|h| format!(" {h}")).unwrap_or_default();
                log::error!("Failed to run `uv sync`: {e}.{hint}");
                let msg = format!("Setup failed: could not run uv.{hint}");
                let _ = app.emit("setup_progress", &msg);
                let _ = std::fs::remove_dir_all(backend_dir.join(".venv"));
                return Err(msg);
            }
        }
    }

    Ok(backend_dir)
}

/// Runs `setup_backend` (which may run `uv sync`) on a blocking thread and
/// returns the resulting directory, or `None` if setup failed — in which
/// case `backend_crashed` has already been emitted.
async fn resolve_backend_dir(
    handle: &tauri::AppHandle,
    uv: &std::path::Path,
) -> Option<std::path::PathBuf> {
    let h = handle.clone();
    let uv_bg = uv.to_path_buf();
    match tokio::task::spawn_blocking(move || setup_backend(&h, &uv_bg)).await {
        Ok(Ok(dir)) => Some(dir),
        Ok(Err(msg)) => {
            let _ = handle.emit("backend_crashed", msg);
            None
        }
        Err(_) => Some(get_backend_dir(handle)),
    }
}

// ── Health polling ────────────────────────────────────────────────────────────

async fn poll_until_ready(port: u16) {
    let url = format!("http://127.0.0.1:{}/health", port);
    let client = reqwest::Client::new();
    loop {
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
}

// ── Backend spawn / monitor ───────────────────────────────────────────────────

async fn spawn_and_monitor_backend(
    handle: tauri::AppHandle,
    port: u16,
    uv: std::path::PathBuf,
    backend_dir: std::path::PathBuf,
    inner: Arc<Mutex<Option<Child>>>,
) {
    let port_str = port.to_string();

    // Choose the Python executable: CUDA venv (AppData) if installed,
    // otherwise the regular venv.  In debug the regular venv is managed
    // by `uv run`; for CUDA we call python directly in both modes so
    // the CUDA packages in AppData are used instead of the source-tree venv.
    let cuda_dir = get_cuda_dir(&handle);
    let python = get_python_exe(&cuda_dir, &backend_dir);
    let using_cuda_venv = python.starts_with(&cuda_dir);

    let mut cmd = if cfg!(debug_assertions) && !using_cuda_venv {
        // Dev without CUDA: let uv manage the venv automatically.
        let mut c = uv_command(&uv);
        c.args(["run", "uvicorn", "live_photo_commentary.main:app",
               "--host", "127.0.0.1", "--port"]);
        c
    } else {
        let mut c = Command::new(&python);
        c.args(["-m", "uvicorn", "live_photo_commentary.main:app",
               "--host", "127.0.0.1", "--port"]);
        // CUDA venv has no editable install of the project; inject the
        // source directory so `live_photo_commentary` is importable.
        if using_cuda_venv {
            c.env("PYTHONPATH", &backend_dir);
        }
        c
    };

    // In release, redirect stdout+stderr to a log file in backend_dir
    // so crashes are diagnosable.
    let log_file = if !cfg!(debug_assertions) {
        std::fs::File::create(backend_dir.join("backend.log")).ok()
    } else {
        None
    };

    // Disable Python's output buffering so backend.log captures crashes that
    // happen before the process has a chance to flush its write buffer.
    cmd.env("PYTHONUNBUFFERED", "1");
    cmd.env("LPC_FRAMES_DIR", backend_dir.join("frames"));

    // Point the backend at the bundled models directory (release only).
    // In debug mode the backend's default ../models already points at the
    // project-root models/ directory that developers edit directly.
    #[cfg(not(debug_assertions))]
    if let Ok(resource_dir) = handle.path().resource_dir() {
        let models_dir = resource_dir.join("resources").join("models");
        if models_dir.exists() {
            cmd.env("MODEL_DIR", models_dir);
        }
    }

    cmd.arg(&port_str)
        .current_dir(&backend_dir)
        .stdout(if cfg!(debug_assertions) {
            Stdio::inherit()
        } else {
            log_file.as_ref()
                .and_then(|f| f.try_clone().ok())
                .map(Stdio::from)
                .unwrap_or_else(Stdio::null)
        })
        .stderr(if cfg!(debug_assertions) {
            Stdio::inherit()
        } else {
            log_file.map(Stdio::from).unwrap_or_else(Stdio::null)
        });

    // Suppress the console window in release builds.
    if !cfg!(debug_assertions) {
        no_window(&mut cmd);
    }

    // Put the child in its own process group so kill() can terminate the whole
    // tree (uv + uvicorn + Python) by sending SIGKILL to the group.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let hint = av_blocked_hint(&e).map(|h| format!(" {h}")).unwrap_or_default();
            let msg = format!("Failed to start backend: {e}.{hint}");
            log::error!("{msg}");
            let _ = handle.emit("backend_crashed", msg);
            return;
        }
    };

    // Insert child into the shared Arc so kill() and the monitor can reach it.
    *inner.lock().unwrap() = Some(child);
    let spawn_time = std::time::Instant::now();

    // ── Crash monitor ────────────────────────────────────────────────────────
    // Polls the child every 500 ms. If the process exits without being killed
    // by us (i.e. it crashed), emits `backend_crashed` so the frontend shows
    // an error instead of retrying forever.
    //
    // Race-free: RunEvent::Exit / Drop and this thread share the same
    // Mutex<Option<Child>>. Whichever locks first takes the Option:
    //  • RunEvent::Exit calls kill() → takes child, sets None.
    //    Monitor's next iteration finds None → exits silently.
    //  • Monitor finds try_wait() returned Some (crashed) → takes it, sets
    //    None, emits event. kill() later finds None → noop.
    let monitor_inner = Arc::clone(&inner);
    let monitor_handle = handle.clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let mut guard = monitor_inner.lock().unwrap();
        match guard.as_mut() {
            None => break, // normal shutdown via Drop
            Some(child) => match child.try_wait() {
                Ok(Some(status)) => {
                    *guard = None; // prevent Drop from double-killing
                    drop(guard);
                    let code = status.code().unwrap_or(-1);
                    let av_hint = quick_exit_hint(spawn_time.elapsed());
                    let log_hint = if cfg!(debug_assertions) { "" } else { " Check backend.log for details." };
                    let msg = format!("Backend exited unexpectedly (code {code}).{av_hint}{log_hint}");
                    log::error!("{msg}");
                    let _ = monitor_handle.emit("backend_crashed", msg);
                    break;
                }
                Ok(None) => {}  // still running
                Err(e) => {
                    log::error!("Error monitoring backend process: {e}");
                    break;
                }
            },
        }
    });

    // Emit `backend_ready` once the health endpoint responds.
    poll_until_ready(port).await;
    let _ = handle.emit("backend_ready", port);
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = find_free_port();

    tauri::Builder::default()
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .setup(move |app| {
            {
                let builder = tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Info);
                #[cfg(not(debug_assertions))]
                let builder = builder.target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::LogDir { file_name: None },
                ));
                app.handle().plugin(builder.build())?;
            }

            if let Some(window) = app.get_webview_window("main") {
                let _ = window.restore_state(tauri_plugin_window_state::StateFlags::all());
            }

            let uv = get_uv_path(app.handle());

            // Manage InstallState immediately — get_backend_dir is a pure path
            // computation and does not run uv sync.
            let backend_dir = get_backend_dir(app.handle());
            app.manage(InstallState {
                uv_path: uv.clone(),
                backend_dir: backend_dir.clone(),
            });

            let frames_dir = backend_dir.join("frames");
            let _ = std::fs::create_dir_all(&frames_dir);
            app.manage(ScreenshotState {
                frames_dir,
                frame_slot: AtomicUsize::new(0),
                is_wsl: detect_wsl(),
            });

            // CUDA upgrade detection — run before managing state so the result
            // is available synchronously when get_backend_port is called.
            #[cfg(not(target_os = "macos"))]
            let cuda_suggestion: Option<String> = {
                let cuda_dir = get_cuda_dir(app.handle());
                let marker = cuda_dir.join(".cuda_torch");
                let installed = std::fs::read_to_string(&marker).unwrap_or_default();
                detect_cuda_index().and_then(|cu_index| {
                    if installed.trim() != cu_index {
                        log::info!("NVIDIA GPU detected, suggesting {cu_index} torch");
                        Some(cu_index.to_string())
                    } else {
                        None
                    }
                })
            };
            #[cfg(target_os = "macos")]
            let cuda_suggestion: Option<String> = None;

            // Pre-allocate BackendState with the chosen port; the child process
            // handle starts as None and is filled in once the background task
            // spawns the backend (after uv sync completes).
            let inner = Arc::new(Mutex::new(None::<Child>));
            let inner_for_bg = Arc::clone(&inner);
            app.manage(BackendState {
                port,
                process: BackendProcess { inner },
                cuda_suggestion,
            });

            // Spawn all heavy work (uv sync, process spawn, health polling) on
            // a background task so setup() returns immediately and the WebView
            // can render the splash screen.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                // setup_backend may run `uv sync`, which can take minutes on
                // first launch — must not run on the event loop thread.
                let backend_dir = match resolve_backend_dir(&handle, &uv).await {
                    Some(dir) => dir,
                    None => return,
                };

                spawn_and_monitor_backend(handle, port, uv, backend_dir, inner_for_bg).await;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_backend_port,
            install_cuda_torch,
            restart_backend,
            list_monitors,
            capture_monitor_preview,
            take_screenshot,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                // Kill the backend here rather than relying solely on Drop.
                // The monitor thread holds an AppHandle clone, which keeps
                // BackendState (and its Drop) alive until the thread exits —
                // but the thread only exits when it sees None in the mutex,
                // which Drop sets... creating a cycle. RunEvent::Exit fires
                // before that cycle becomes a problem.
                app_handle.state::<BackendState>().process.kill();
            }
        });
}
