use std::process::Child;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use tauri::Manager;
use tauri_plugin_window_state::WindowExt;

mod antivirus;
mod backend;
mod cuda;
mod flash_attn;
mod screensaver;
mod screenshot;
mod util;
mod uv;

use backend::{BackendProcess, BackendState};
use cuda::InstallState;
use screenshot::ScreenshotState;

/// Installs the tauri-plugin-log logger. Release builds also write to a file in
/// the platform log directory.
fn init_logging(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let builder = tauri_plugin_log::Builder::default().level(log::LevelFilter::Info);
    #[cfg(not(debug_assertions))]
    let builder = builder.target(tauri_plugin_log::Target::new(
        tauri_plugin_log::TargetKind::LogDir { file_name: None },
    ));
    app.handle().plugin(builder.build())?;
    Ok(())
}

/// Restores the saved window geometry, then recenters the window on the primary
/// monitor if the restored position lies entirely off every connected monitor
/// (e.g. the monitor it was on has since been disconnected).
fn restore_window(app: &tauri::App) {
    let Some(window) = app.get_webview_window("main") else { return };
    let _ = window.restore_state(tauri_plugin_window_state::StateFlags::all());

    if let (Ok(pos), Ok(size)) = (window.outer_position(), window.outer_size()) {
        let monitors = app.available_monitors().unwrap_or_default();
        let on_screen = monitors.iter().any(|m| {
            let mp = m.position();
            let ms = m.size();
            pos.x + size.width as i32 > mp.x
                && pos.x < mp.x + ms.width as i32
                && pos.y + size.height as i32 > mp.y
                && pos.y < mp.y + ms.height as i32
        });
        if !on_screen {
            let _ = window.center();
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = util::find_free_port();

    // Register signal handlers before spawning anything so Ctrl-C and SIGTERM
    // always kill the backend process group before terminating.
    #[cfg(unix)]
    backend::install_signal_handlers();

    tauri::Builder::default()
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .setup(move |app| {
            init_logging(app)?;
            restore_window(app);

            let uv = uv::get_uv_path(app.handle());

            // Manage InstallState immediately — get_backend_dir is a pure path
            // computation and does not run uv sync.
            let backend_dir = backend::get_backend_dir(app.handle());
            app.manage(InstallState {
                uv_path: uv.clone(),
                backend_dir: backend_dir.clone(),
            });

            let frames_dir = backend_dir.join("frames");
            let _ = std::fs::create_dir_all(&frames_dir);
            app.manage(ScreenshotState {
                frames_dir,
                frame_slot: AtomicUsize::new(0),
                is_wsl: screenshot::detect_wsl(),
            });

            // Run before managing state so the result is available synchronously
            // when get_backend_port is called.
            let cuda_suggestion = cuda::detect_cuda_suggestion(app.handle());

            // Pre-allocate BackendState with the chosen port; the child process
            // handle starts as None and is filled in once the background task
            // spawns the backend (after uv sync completes).
            let inner = Arc::new(Mutex::new(None::<Child>));
            app.manage(BackendState {
                port,
                process: BackendProcess { inner: Arc::clone(&inner) },
                cuda_suggestion,
            });
            app.manage(screensaver::ScreensaverState::default());

            backend::spawn_backend_setup(app.handle().clone(), port, uv, inner);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            backend::get_backend_port,
            cuda::install_cuda_torch,
            backend::restart_backend,
            screenshot::list_monitors,
            screenshot::capture_monitor_preview,
            screenshot::take_screenshot,
            screensaver::inhibit_screensaver,
            screensaver::allow_screensaver,
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
