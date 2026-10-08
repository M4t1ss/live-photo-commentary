use std::sync::Arc;
use tauri::Manager;
use tauri_plugin_window_state::WindowExt;

// The Python backend's supervision code. Nothing starts it any more (phase 5);
// `AppState::new` still uses `backend::get_backend_dir` to find the old data
// folders to migrate. All removed in phase 7.
#[allow(dead_code)]
mod antivirus;
#[allow(dead_code)]
mod backend;
#[allow(dead_code)]
mod cuda;
#[allow(dead_code)]
mod flash_attn;
#[allow(dead_code)]
mod util;
#[allow(dead_code)]
mod uv;

mod avatar;
mod catalogue;
mod commands;
mod config;
mod cuda_backend;
mod difference;
mod events;
mod gpu;
mod pipeline;
mod promptsets;
mod screensaver;
mod screenshot;
mod state;

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
    tauri::Builder::default()
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .setup(|app| {
            init_logging(app)?;
            restore_window(app);

            let frames_dir = app.path().app_data_dir()?.join("frames");
            let _ = std::fs::create_dir_all(&frames_dir);
            app.manage(ScreenshotState {
                frames_dir,
                is_wsl: screenshot::detect_wsl(),
            });
            app.manage(screensaver::ScreensaverState::default());

            gpu::init(app.handle());
            let app_state = Arc::new(state::AppState::new(app.handle())?);
            let scope = app.asset_protocol_scope();
            let _ = scope.allow_directory(app_state.model_dir(), true);
            let _ = scope.allow_directory(app_state.avatar_models().user_dir(), true);
            let _ = scope.allow_directory(app_state.animations_dir(), true);
            app.manage(Arc::clone(&app_state));
            tauri::async_runtime::spawn(async move { app_state.init_models().await });

            Ok(())
        })
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            screenshot::list_monitors,
            screenshot::capture_monitor_preview,
            screenshot::take_screenshot,
            screensaver::inhibit_screensaver,
            screensaver::allow_screensaver,
            commands::connect,
            commands::get_config,
            commands::set_config,
            commands::get_models,
            commands::start_cycle,
            commands::stop_cycle,
            commands::list_promptsets,
            commands::apply_prompts,
            commands::load_promptset,
            commands::save_promptset,
            commands::delete_promptset,
            commands::synthesize,
            commands::play_system_message,
            commands::model_config,
            commands::avatar_models,
            commands::open_model_dir,
            commands::upload_model,
            commands::gpu_status,
            commands::download_cuda_backend,
            commands::remove_cuda_backend,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_, _| {});
}
