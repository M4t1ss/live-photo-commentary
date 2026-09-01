//! Monitor enumeration and desktop screenshot capture.
//!
//! Native capture uses the `screenshots` crate. When running inside WSL that
//! can't reach the Windows desktop, so we shell out to a cross-compiled
//! `screenshot.exe` helper instead (see `src-screenshot/`).

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use tauri::{Manager, State};

pub(crate) struct ScreenshotState {
    pub(crate) frames_dir: std::path::PathBuf,
    pub(crate) frame_slot: AtomicUsize,
    pub(crate) is_wsl: bool,
}

#[derive(serde::Serialize)]
pub(crate) struct MonitorInfo {
    index: usize,
    name: String,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    is_primary: bool,
}

/// True when running inside WSL — detected via the `microsoft` marker in
/// `/proc/version`. Used to decide between native capture and `screenshot.exe`.
pub(crate) fn detect_wsl() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|v| v.to_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

fn png_to_data_url(path: &std::path::Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read failed: {e}"))?;
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:image/png;base64,{b64}"))
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

#[tauri::command]
pub fn list_monitors() -> Result<Vec<MonitorInfo>, String> {
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
pub fn capture_monitor_preview(
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
pub fn take_screenshot(
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
