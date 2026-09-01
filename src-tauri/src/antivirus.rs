//! Windows Defender / antivirus interference detection and mitigation.
//!
//! On Windows, Defender can block a just-installed executable from running,
//! terminate a process the instant it spawns, or corrupt a managed-Python
//! extraction mid-write. These helpers recognise those failure modes so the
//! caller can retry, quarantine the broken install, or surface a useful hint.

/// Returns a hint string when an IO error looks like Windows Defender or
/// antivirus software blocked an executable from running.
pub(crate) fn av_blocked_hint(e: &std::io::Error) -> Option<&'static str> {
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

/// How long to keep retrying an operation that fails in a way `av_blocked_hint`
/// recognizes — e.g. `uv sync` or spawning a just-installed executable while
/// Defender is still scanning it. One shared constant so every retry site
/// waits the same amount before giving up.
const AV_RETRY_WINDOW: std::time::Duration = std::time::Duration::from_secs(5);

/// Retries `f` while it fails in a way `av_blocked_hint` recognizes as
/// Defender/AV interference, for up to `AV_RETRY_WINDOW`. Any other error, or
/// success, returns immediately — this never masks a real failure.
pub(crate) fn retry_while_av_blocked<T>(
    mut f: impl FnMut() -> std::io::Result<T>,
) -> std::io::Result<T> {
    let start = std::time::Instant::now();
    loop {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) if av_blocked_hint(&e).is_some() && start.elapsed() < AV_RETRY_WINDOW => {
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            Err(e) => return Err(e),
        }
    }
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

/// Quarantines broken uv-managed Python installs under `%LOCALAPPDATA%\uv\python`.
/// No-op on non-Windows platforms.
pub(crate) fn quarantine_local_python() {
    #[cfg(target_os = "windows")]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let py_dir = std::path::Path::new(&local).join("uv").join("python");
        quarantine_broken_python_installs(&py_dir);
    }
}

/// Returns a hint string when a process exits very shortly after spawning.
/// An immediate exit is a common pattern when AV software terminates a
/// process as it launches.
pub(crate) fn quick_exit_hint(elapsed: std::time::Duration) -> &'static str {
    #[cfg(target_os = "windows")]
    if elapsed < std::time::Duration::from_secs(3) {
        return " Windows Defender or antivirus may have terminated the process immediately. \
                Open Windows Security → Virus & threat protection → Protection history \
                to check, or add the app data folder to Defender exclusions and restart.";
    }
    let _ = elapsed;
    ""
}
