use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager, State};

// ── App state ────────────────────────────────────────────────────────────────

struct BackendState {
    port: u16,
    process: BackendProcess,
}

/// Wraps the child process handle so we can share it between the shutdown handler
/// (which kills on normal exit), the Drop impl (safety net), and the monitor
/// thread (which detects crashes).
struct BackendProcess {
    inner: Arc<Mutex<Option<Child>>>,
}

impl BackendProcess {
    /// Kills the backend process (and its children on Windows).
    /// Safe to call multiple times — the Option is taken on first call;
    /// subsequent calls find None and return immediately.
    fn kill(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(mut child) = guard.take() {
                #[cfg(target_os = "windows")]
                {
                    use std::os::windows::process::CommandExt;
                    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                    let _ = Command::new("taskkill")
                        .args(["/F", "/T", "/PID", &child.id().to_string()])
                        .creation_flags(CREATE_NO_WINDOW)
                        .status();
                }
                #[cfg(not(target_os = "windows"))]
                let _ = child.kill();
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

// ── Tauri command ─────────────────────────────────────────────────────────────

#[tauri::command]
fn get_backend_port(state: State<BackendState>) -> u16 {
    state.port
}

// ── Port discovery ────────────────────────────────────────────────────────────

fn find_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
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
/// Blocks until complete — the window will be delayed on first launch.
/// TODO: Show a "Setting up…" progress window during this step.
fn setup_backend(app: &tauri::AppHandle, uv: &std::path::Path) -> std::path::PathBuf {
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
                if let Err(e) = copy_dir_recursive(&resource_backend, &backend_dir) {
                    log::error!("Failed to copy backend source: {e}");
                    return backend_dir;
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
                return backend_dir;
            }
        }
    }

    // Run `uv sync` if the venv is absent or incomplete.
    // Checking pyvenv.cfg (not just the .venv dir) catches partial installs
    // where the directory was created but uv sync didn't finish.
    if !backend_dir.join(".venv").join("pyvenv.cfg").exists() {
        log::info!("Running `uv sync` in {} …", backend_dir.display());

        let mut sync_cmd = Command::new(uv);
        sync_cmd.arg("sync").current_dir(&backend_dir);

        // In release: suppress console window on Windows, redirect output to a
        // log file so failures are diagnosable without attaching a debugger.
        #[cfg(target_os = "windows")]
        if !cfg!(debug_assertions) {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            sync_cmd.creation_flags(CREATE_NO_WINDOW);
        }
        if !cfg!(debug_assertions) {
            if let Ok(log_file) = std::fs::File::create(backend_dir.join("uv-sync.log")) {
                if let Ok(log_clone) = log_file.try_clone() {
                    sync_cmd
                        .stdout(Stdio::from(log_file))
                        .stderr(Stdio::from(log_clone));
                }
            }
        }

        match sync_cmd.status() {
            Ok(s) if s.success() => log::info!("Python environment ready."),
            Ok(s) => log::error!("`uv sync` failed (exit code {:?})", s.code()),
            Err(e) => log::error!("Failed to run `uv sync`: {e}"),
        }
    }

    backend_dir
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

// ── Entry point ───────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = find_free_port();

    tauri::Builder::default()
        .setup(move |app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            let uv = get_uv_path(app.handle());
            let backend_dir = setup_backend(app.handle(), &uv);

            let port_str = port.to_string();

            // In debug: use `uv run` so the dev venv is auto-managed.
            // In release: call the venv Python directly — uv sync has already
            // set it up, and bypassing `uv run` avoids a console flash on
            // Windows caused by uv spawning its own subprocess internally.
            let mut cmd = if cfg!(debug_assertions) {
                let mut c = Command::new(&uv);
                c.args(["run", "uvicorn", "live_photo_commentary.main:app", "--host", "127.0.0.1", "--port"]);
                c
            } else {
                let python = if cfg!(target_os = "windows") {
                    backend_dir.join(".venv").join("Scripts").join("python.exe")
                } else {
                    backend_dir.join(".venv").join("bin").join("python")
                };
                let mut c = Command::new(python);
                c.args(["-m", "uvicorn", "live_photo_commentary.main:app", "--host", "127.0.0.1", "--port"]);
                c
            };
            // In release, redirect stdout+stderr to a log file in backend_dir
            // so crashes are diagnosable. Falls back to null if the file can't
            // be created (e.g. backend_dir doesn't exist yet).
            let log_file = if !cfg!(debug_assertions) {
                std::fs::File::create(backend_dir.join("backend.log")).ok()
            } else {
                None
            };

            cmd.arg(&port_str)
                .current_dir(&backend_dir)
                .stdout(if cfg!(debug_assertions) {
                    Stdio::inherit()
                } else {
                    log_file.as_ref().and_then(|f| f.try_clone().ok()).map(Stdio::from).unwrap_or_else(Stdio::null)
                })
                .stderr(if cfg!(debug_assertions) {
                    Stdio::inherit()
                } else {
                    log_file.map(Stdio::from).unwrap_or_else(Stdio::null)
                });

            // On Windows release builds, set CREATE_NO_WINDOW so the OS
            // doesn't open a console window for the child process.
            #[cfg(target_os = "windows")]
            if !cfg!(debug_assertions) {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                cmd.creation_flags(CREATE_NO_WINDOW);
            }

            let child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    log::error!("Failed to spawn backend process: {e}");
                    // Emit the crash event so the frontend shows an error
                    // rather than spinning on a health check that never resolves.
                    let _ = app.handle().emit("backend_crashed", -1i32);
                    return Ok(());
                }
            };

            let inner = Arc::new(Mutex::new(Some(child)));

            // ── Crash monitor ────────────────────────────────────────────────
            // Polls the child every 500 ms. If the process exits without being
            // killed by us (i.e. it crashed), emits `backend_crashed` to the
            // frontend so it can show an error instead of retrying forever.
            //
            // Race-free: RunEvent::Exit / Drop and this thread share the same
            // Mutex<Option<Child>>. Whichever locks first takes the Option:
            //  • RunEvent::Exit calls kill() → takes child, sets None.
            //    Monitor's next iteration finds None → exits silently.
            //  • Monitor finds try_wait() returned Some (crashed) → takes it,
            //    sets to None, emits event.
            //    kill() later finds None → does nothing.
            let monitor_inner = Arc::clone(&inner);
            let monitor_handle = app.handle().clone();
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
                            log::error!("Backend exited unexpectedly (code {code})");
                            let _ = monitor_handle.emit("backend_crashed", code);
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

            app.manage(BackendState {
                port,
                process: BackendProcess { inner },
            });

            // Emit `backend_ready` once the health endpoint responds.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                poll_until_ready(port).await;
                let _ = handle.emit("backend_ready", port);
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_backend_port])
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
