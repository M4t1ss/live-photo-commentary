//! Monitor enumeration and desktop screenshot capture.
//!
//! Native capture uses the `xcap` crate. When running inside WSL that
//! can't reach the Windows desktop, so we shell out to a cross-compiled
//! `screenshot.exe` helper instead (see `src-screenshot/`).

use std::process::Command;
use tauri::{Manager, State};

pub(crate) struct ScreenshotState {
    /// Where `screenshot.exe` (WSL only) writes its PNGs. The native paths
    /// never touch the disk.
    pub(crate) frames_dir: std::path::PathBuf,
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
            .expect("CARGO_MANIFEST_DIR should have a parent")
            .join("screenshot.exe")
    } else {
        app.path()
            .resource_dir()
            .expect("resource dir should be available")
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
    let info = |i, m: &xcap::Monitor| -> xcap::XCapResult<MonitorInfo> {
        Ok(MonitorInfo {
            index: i,
            name: format!("Display {}", m.id()?),
            x: m.x()?,
            y: m.y()?,
            width: m.width()?,
            height: m.height()?,
            is_primary: m.is_primary()?,
        })
    };
    let monitors = xcap::Monitor::all().map_err(|e| format!("screen enumeration failed: {e}"))?;
    monitors
        .iter()
        .enumerate()
        .map(|(i, m)| info(i, m).map_err(|e| format!("monitor info failed: {e}")))
        .collect()
}

/// Captures monitor `idx` (the first one if there's no such monitor).
fn capture_monitor(idx: Option<u32>) -> Result<image::RgbaImage, String> {
    let monitors = xcap::Monitor::all().map_err(|e| format!("screen enumeration failed: {e}"))?;
    let idx = idx.unwrap_or(0) as usize;
    let monitor = monitors.get(idx).or_else(|| monitors.first()).ok_or("no screens found")?;
    monitor.capture_image().map_err(|e| format!("capture failed: {e}"))
}

/// Async for the same reason as `take_screenshot`: the capture and PNG
/// encoding run on a blocking-pool thread, not the main (UI) thread.
#[tauri::command]
pub async fn capture_monitor_preview(
    app: tauri::AppHandle,
    state: State<'_, ScreenshotState>,
    monitor_idx: Option<u32>,
) -> Result<String, String> {
    let preview_path = state.frames_dir.join("frame_preview.png");
    let is_wsl = state.is_wsl;
    tauri::async_runtime::spawn_blocking(move || {
        if is_wsl {
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
            return png_to_data_url(&preview_path);
        }
        let image = image::DynamicImage::ImageRgba8(capture_monitor(monitor_idx)?);
        Ok(crate::events::png_data_url(&image))
    })
    .await
    .map_err(|e| format!("preview task failed: {e}"))?
}

/// Captures a frame and hands it straight to the pipeline
/// (`Pipeline::trigger`); the frontend no longer sends `frame_ready` over a
/// WebSocket. Ignored while the cycle isn't running, like Python's frame
/// queue.
///
/// Natively the frame stays in memory from capture to the channel. Only the
/// WSL helper, `screenshot.exe`, can't return an image, so it writes a file
/// that is read back.
///
/// Async so the work (capture, PNG encoding, frame differencing) runs on a
/// blocking-pool thread: a synchronous command would run on the main (UI)
/// thread.
#[tauri::command]
pub async fn take_screenshot(
    app: tauri::AppHandle,
    state: State<'_, ScreenshotState>,
    app_state: State<'_, std::sync::Arc<crate::state::AppState>>,
    monitor_idx: Option<u32>,
    clip: Option<[i32; 4]>,
) -> Result<(), String> {
    if !app_state.pipeline().running() {
        return Ok(());
    }
    let wsl_file = state.frames_dir.join("frame_wsl.png");
    let is_wsl = state.is_wsl;
    let app_state = std::sync::Arc::clone(&app_state);

    tauri::async_runtime::spawn_blocking(move || {
        let result = (|| -> Result<image::DynamicImage, String> {
            if is_wsl {
                let exe = locate_screenshot_exe(&app);
                let win_dest = wslpath_to_windows(&wsl_file)?;
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
                image::open(&wsl_file).map_err(|e| format!("reload failed: {e}"))
            } else {
                let img = capture_monitor(monitor_idx)?;
                let img = if let Some([x, y, w, h]) = clip {
                    let ix = x.max(0) as u32;
                    let iy = y.max(0) as u32;
                    let iw = (w as u32).min(img.width().saturating_sub(ix)).max(1);
                    let ih = (h as u32).min(img.height().saturating_sub(iy)).max(1);
                    image::DynamicImage::ImageRgba8(img).crop_imm(ix, iy, iw, ih).into_rgba8()
                } else {
                    img
                };
                Ok(image::DynamicImage::ImageRgba8(img))
            }
        })();

        match result {
            Ok(image) => {
                // The cycle may have been stopped while the capture ran.
                if app_state.pipeline().running() {
                    app_state.pipeline().trigger(&app_state.config(), image);
                }
                Ok(())
            }
            Err(msg) => {
                log::error!("take_screenshot failed: {msg}");
                Err(msg)
            }
        }
    })
    .await
    .map_err(|e| format!("screenshot task failed: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Times each step of `take_screenshot` on a real capture of monitor 0.
    /// Needs a display, so it's ignored by default:
    /// `cargo test -p app --lib -- --ignored --nocapture time_screenshot_steps`
    #[test]
    #[ignore]
    fn time_screenshot_steps() {
        for round in 0..4 {
            let t = Instant::now();
            let img = capture_monitor(Some(0)).expect("capture");
            let capture = t.elapsed();

            let (w, h) = (img.width(), img.height());
            let image = std::sync::Arc::new(image::DynamicImage::ImageRgba8(img));

            let t = Instant::now();
            let json = crate::events::Event::Frame { image: image.clone(), push: true, gen: 1, diff: None, measure: None }.to_json();
            let frame_event = t.elapsed();

            let t = Instant::now();
            let text = serde_json::to_string(&json).expect("serialize");
            let serialize = t.elapsed();

            // For comparison: the same encode with the fast PNG settings.
            let t = Instant::now();
            let mut fast = std::io::Cursor::new(Vec::new());
            let encoder = image::codecs::png::PngEncoder::new_with_quality(
                &mut fast,
                image::codecs::png::CompressionType::Fast,
                image::codecs::png::FilterType::Sub,
            );
            image.write_with_encoder(encoder).expect("fast encode");
            let fast_encode = t.elapsed();

            eprintln!(
                "round {round}: {w}x{h}  capture {capture:?} | Frame event to_json {frame_event:?} \
                 (JSON text {} KB, serialize {serialize:?}) | fast PNG encode {fast_encode:?} ({} KB)",
                text.len() / 1024,
                fast.get_ref().len() / 1024,
            );
        }
    }
}
