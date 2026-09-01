//! Lifecycle of the Python FastAPI sidecar: environment setup (`uv sync`),
//! spawning uvicorn, crash monitoring, health polling, and teardown.

use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
#[cfg(unix)]
use std::sync::atomic::{AtomicI32, Ordering};
use tauri::{Emitter, Manager, State};

use crate::antivirus::{
    av_blocked_hint, quarantine_local_python, quick_exit_hint, retry_while_av_blocked,
};
use crate::cuda::{get_cuda_dir, get_python_exe, InstallState};
use crate::util::{log_to_file, no_window};
use crate::uv::uv_command;

// ── Signal handling (Unix only) ───────────────────────────────────────────────
//
// When `cargo tauri dev` is stopped with Ctrl-C, SIGINT is delivered to the
// entire terminal process group, which includes the Tauri binary. The default
// SIGINT action terminates the process immediately — no destructors, no
// RunEvent::Exit — so the backend (in its own process group) is orphaned.
//
// Fix: intercept SIGINT/SIGTERM, kill the backend's process group first
// (async-signal-safe: only uses kill/signal/raise), then re-raise so the
// process exits with the correct signal disposition.

#[cfg(unix)]
static BACKEND_PGID: AtomicI32 = AtomicI32::new(-1);

#[cfg(unix)]
extern "C" fn on_term_signal(sig: libc::c_int) {
    let pgid = BACKEND_PGID.load(Ordering::Relaxed);
    if pgid > 0 {
        unsafe { libc::kill(-(pgid as libc::pid_t), libc::SIGKILL); }
    }
    // Reset to default and re-raise so callers see the correct exit status.
    unsafe {
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

/// Intercepts SIGINT/SIGTERM so the backend process group is killed before the
/// Tauri binary exits. Must run before the backend is spawned.
#[cfg(unix)]
pub(crate) fn install_signal_handlers() {
    unsafe {
        libc::signal(libc::SIGINT, on_term_signal as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_term_signal as libc::sighandler_t);
    }
}

// ── App state ────────────────────────────────────────────────────────────────

pub(crate) struct BackendState {
    pub(crate) port: u16,
    pub(crate) process: BackendProcess,
    pub(crate) cuda_suggestion: Option<String>,
}

/// Wraps the child process handle so we can share it between the shutdown handler
/// (which kills on normal exit), the Drop impl (safety net), and the monitor
/// thread (which detects crashes).
pub(crate) struct BackendProcess {
    pub(crate) inner: Arc<Mutex<Option<Child>>>,
}

impl BackendProcess {
    /// Kills the backend process and its children.
    /// Safe to call multiple times — the Option is taken on first call;
    /// subsequent calls find None and return immediately.
    pub(crate) fn kill(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(mut child) = guard.take() {
                // Clear the global PGID before killing so a concurrent signal
                // handler won't try to kill the same group twice.
                #[cfg(unix)]
                BACKEND_PGID.store(-1, Ordering::Relaxed);

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

// ── Tauri commands ───────────────────────────────────────────────────────────

#[tauri::command]
pub fn get_backend_port(state: State<BackendState>) -> (u16, Option<String>) {
    (state.port, state.cuda_suggestion.clone())
}

#[tauri::command]
pub fn restart_backend(
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

// ── Backend directory & setup ────────────────────────────────────────────────

/// Returns the writable backend directory for the current mode:
/// - Debug: the source tree (`backend/` next to `src-tauri/`)
/// - Release: `<app_data_dir>/backend/` — always writable, so `uv sync` can
///   create `.venv` there even on read-only install paths (Program Files, .app)
pub(crate) fn get_backend_dir(app: &tauri::AppHandle) -> std::path::PathBuf {
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

        match retry_while_av_blocked(|| sync_cmd.status()) {
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

// ── Health polling ───────────────────────────────────────────────────────────

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

// ── Backend spawn / monitor ──────────────────────────────────────────────────

/// Spawns backend setup + monitoring on a background task so `setup()` returns
/// immediately and the WebView can render the splash screen. `setup_backend`
/// may run `uv sync`, which can take minutes on first launch — it must never
/// run on the event loop thread.
pub(crate) fn spawn_backend_setup(
    handle: tauri::AppHandle,
    port: u16,
    uv: std::path::PathBuf,
    inner: Arc<Mutex<Option<Child>>>,
) {
    tauri::async_runtime::spawn(async move {
        let Some(backend_dir) = resolve_backend_dir(&handle, &uv).await else {
            return;
        };
        spawn_and_monitor_backend(handle, port, uv, backend_dir, inner).await;
    });
}

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
    if cfg!(debug_assertions) {
        cmd.env("LPC_DEV", "1");
    }

    // Let the backend lazily install the Japanese TTS tokenizer (Sudachi + its
    // large dictionary) via uv the first time a Japanese voice is selected.
    // Only the Rust side knows where the uv binary and the live venv are.
    cmd.env("LPC_UV", &uv);
    cmd.env("LPC_TARGET_PYTHON", &python);
    if using_cuda_venv {
        cmd.env("LPC_CUDA_VENV", "1");
    }

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

    // python.exe (or its DLLs) may have just been (re)installed by
    // install_cuda_torch / setup_backend and still be getting scanned by
    // Defender — retry briefly rather than surfacing a spurious crash.
    let child = match retry_while_av_blocked(|| cmd.spawn()) {
        Ok(c) => c,
        Err(e) => {
            let hint = av_blocked_hint(&e).map(|h| format!(" {h}")).unwrap_or_default();
            let msg = format!("Failed to start backend: {e}.{hint}");
            log::error!("{msg}");
            let _ = handle.emit("backend_crashed", msg);
            return;
        }
    };

    // Publish PGID for the signal handler before inserting into the mutex.
    // process_group(0) means the child's PGID equals its PID.
    #[cfg(unix)]
    BACKEND_PGID.store(child.id() as i32, Ordering::Relaxed);

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
