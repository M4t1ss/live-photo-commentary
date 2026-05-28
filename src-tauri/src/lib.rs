use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager, State};

// ── App state ────────────────────────────────────────────────────────────────

struct BackendState {
    port: u16,
    #[allow(dead_code)] // held for Drop, which kills the child process on exit
    process: BackendProcess,
}

/// Wraps the child process handle so we can share it between the Drop impl
/// (which kills on normal shutdown) and the monitor thread (which detects crashes).
struct BackendProcess {
    inner: Arc<Mutex<Option<Child>>>,
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(mut child) = guard.take() {
                // On Windows, `uv run` spawns Python as a child of uv.
                // Killing only uv leaves the Python/uvicorn process running,
                // which keeps .venv files locked. `taskkill /F /T` kills the
                // entire process tree before we wait on the direct child.
                #[cfg(target_os = "windows")]
                {
                    // Kill the whole process tree (uv + its Python child).
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
            // If guard holds None, the monitor thread already cleaned up (crash path).
        }
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
///   1. If `<app_data_dir>/backend/` doesn't exist yet, copies the bundled
///      source from `<resource_dir>/resources/backend/` (read-only) to the
///      writable app data location.
///   2. Runs `uv sync` if `.venv` is absent (first launch or fresh copy).
///
/// Blocks until complete — the window will be delayed on first launch.
/// TODO: Show a "Setting up…" progress window during this step.
fn setup_backend(app: &tauri::AppHandle, uv: &std::path::Path) -> std::path::PathBuf {
    let backend_dir = get_backend_dir(app);

    if !cfg!(debug_assertions) && !backend_dir.exists() {
        let resource_backend = app
            .path()
            .resource_dir()
            .expect("resource dir unavailable")
            .join("resources")
            .join("backend");

        if resource_backend.exists() {
            log::info!(
                "First launch: copying backend source to {} …",
                backend_dir.display()
            );
            if let Err(e) = copy_dir_recursive(&resource_backend, &backend_dir) {
                log::error!("Failed to copy backend source: {e}");
                return backend_dir;
            }
        } else {
            log::error!(
                "Bundled backend not found at {} — backend will not start",
                resource_backend.display()
            );
            return backend_dir;
        }
    }

    // Run `uv sync` if the venv doesn't exist yet.
    if !backend_dir.join(".venv").exists() {
        log::info!("Running `uv sync` in {} …", backend_dir.display());
        match Command::new(uv).arg("sync").current_dir(&backend_dir).status() {
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
            let mut cmd = Command::new(&uv);
            cmd.args(["run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port"])
                .arg(&port_str)
                .current_dir(&backend_dir)
                // In debug, inherit so uvicorn output appears in the terminal.
                // In release, suppress — the GUI app has no console, and we
                // don't want a stray terminal window popping up on Windows.
                .stdout(if cfg!(debug_assertions) { Stdio::inherit() } else { Stdio::null() })
                .stderr(if cfg!(debug_assertions) { Stdio::inherit() } else { Stdio::null() });

            // On Windows release builds, set CREATE_NO_WINDOW so the OS
            // doesn't open a console window for the child process.
            #[cfg(target_os = "windows")]
            if !cfg!(debug_assertions) {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                cmd.creation_flags(CREATE_NO_WINDOW);
            }

            let child = cmd
                .spawn()
                .expect("failed to spawn backend — is `uv` in PATH?");

            let inner = Arc::new(Mutex::new(Some(child)));

            // ── Crash monitor ────────────────────────────────────────────────
            // Polls the child every 500 ms. If the process exits without being
            // killed by Drop (i.e. it crashed), emits `backend_crashed` to the
            // frontend so it can show an error instead of retrying forever.
            //
            // Race-free: Drop and this thread share the same Mutex<Option<Child>>.
            // Whichever locks first takes the Option:
            //  • Drop takes it → kills + waits, sets to None.
            //    Monitor's next iteration finds None → exits silently.
            //  • Monitor finds try_wait() returned Some (crashed) → takes it,
            //    sets to None, emits event.
            //    Drop later finds None → does nothing.
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
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
