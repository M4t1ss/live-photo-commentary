//! Locating and invoking the bundled (or system) `uv` binary.

use std::process::Command;
use tauri::Manager;

/// Builds a `Command` for the `uv` binary at `uv`.
///
/// On Windows, AppData\Roaming is often redirected by OneDrive's Known Folder
/// Move, which creates reparse points that Windows refuses to traverse (error
/// 448). uv defaults to installing managed Pythons under AppData\Roaming\uv,
/// so redirect it to AppData\Local\uv which OneDrive does not touch.
pub(crate) fn uv_command(uv: &std::path::Path) -> Command {
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

/// Returns the path to the uv binary.
///
/// Production: `tauri.conf.json` bundles `resources/uv-<triple>` via the glob
/// `"resources/uv-*"`. With Tauri's array resource format the file lands at
/// `resource_dir()/resources/uv-<triple>`, so we probe that platform-specific
/// name at runtime.
///
/// Development / fallback: the resources directory won't contain the binary
/// (it's only downloaded by CI), so we fall back to whichever `uv` is on PATH.
pub(crate) fn get_uv_path(app: &tauri::AppHandle) -> std::path::PathBuf {
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
