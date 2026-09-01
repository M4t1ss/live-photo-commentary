//! Small cross-cutting helpers with no more specific home.

use std::process::{Command, Stdio};

/// Picks a currently-free TCP port on localhost by binding `127.0.0.1:0`,
/// reading back the OS-assigned port, and dropping the listener.
pub(crate) fn find_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Suppresses the console window for a spawned child process on Windows.
/// No-op on other platforms.
pub(crate) fn no_window(_cmd: &mut Command) {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        _cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

/// Opens `path` as a combined stdout+stderr log file.
/// Falls back to null handles if the file cannot be created.
pub(crate) fn log_to_file(path: &std::path::Path) -> (Stdio, Stdio) {
    std::fs::File::create(path)
        .ok()
        .and_then(|f| f.try_clone().ok().map(|c| (Stdio::from(f), Stdio::from(c))))
        .unwrap_or_else(|| (Stdio::null(), Stdio::null()))
}

/// Windows only: force the just-`uv sync`ed venv to settle before the backend
/// imports torch.
///
/// Right after `uv sync` populates a venv (thousands of hardlinks, ~4 GB of
/// CUDA DLLs), two things lag for a short window: NTFS's per-directory index
/// (so an enumeration can miss a file that is actually on disk — e.g.
/// `torch/nn/__init__.py`, which surfaces as a bogus "partially initialized
/// module 'torch'" ImportError) and Windows Defender's on-access scan (the
/// first `LoadLibrary` of a fresh DLL stalls or faults while it is scanned).
/// The failed first `import torch` then leaves torch's native module
/// half-initialized and the interpreter access-violates on shutdown.
///
/// One synchronous pass here — recursively enumerate every directory, `stat`
/// every file, and fully read every native module (`.dll`/`.pyd`/`.exe`) —
/// reconciles the index and primes the AV scan cache, so the backend's first
/// import is clean. A `.warmed` marker keyed to `site-packages`' mtime makes
/// this a no-op on ordinary launches (no `uv sync`, nothing changed).
/// No-op off Windows.
#[cfg(not(target_os = "windows"))]
pub(crate) fn warm_venv(_venv_dir: &std::path::Path) {}

#[cfg(target_os = "windows")]
pub(crate) fn warm_venv(venv_dir: &std::path::Path) {
    let site_packages = venv_dir.join("Lib").join("site-packages");
    if !site_packages.is_dir() {
        return;
    }

    let marker = venv_dir.join(".warmed");
    let sp_mtime = std::fs::metadata(&site_packages).and_then(|m| m.modified()).ok();
    let marker_mtime = std::fs::metadata(&marker).and_then(|m| m.modified()).ok();
    if let (Some(sp), Some(mk)) = (sp_mtime, marker_mtime) {
        if mk >= sp {
            return; // venv unchanged since the last warm-up
        }
    }

    let start = std::time::Instant::now();
    let (mut dirs, mut files, mut bytes, mut errors) = (0u64, 0u64, 0u64, 0u64);
    let mut stack = vec![site_packages.clone(), venv_dir.join("Scripts")];
    while let Some(dir) = stack.pop() {
        let rd = match retry_io(|| std::fs::read_dir(&dir)) {
            Ok(rd) => rd,
            Err(_) => {
                errors += 1;
                continue;
            }
        };
        dirs += 1;
        for entry in rd {
            let Ok(entry) = entry else {
                errors += 1;
                continue;
            };
            let path = entry.path();
            match entry.file_type() {
                Ok(t) if t.is_dir() => {
                    stack.push(path);
                    continue;
                }
                Ok(_) => {}
                Err(_) => {
                    errors += 1;
                    continue;
                }
            }
            files += 1;
            let _ = std::fs::metadata(&path); // touch the index entry
            let is_native = matches!(
                path.extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_ascii_lowercase)
                    .as_deref(),
                Some("dll") | Some("pyd") | Some("exe")
            );
            if is_native {
                match retry_io(|| {
                    let mut f = std::fs::File::open(&path)?;
                    std::io::copy(&mut f, &mut std::io::sink())
                }) {
                    Ok(n) => bytes += n,
                    Err(_) => errors += 1,
                }
            }
        }
    }

    let _ = std::fs::File::create(&marker);
    log::info!(
        "warm_venv({}): {} dirs, {} files, {:.0} MiB read, {} errors in {:.1}s",
        venv_dir.display(),
        dirs,
        files,
        bytes as f64 / 1_048_576.0,
        errors,
        start.elapsed().as_secs_f64(),
    );
}

/// Retries a filesystem op a few times with a short pause — covers the brief
/// window where a just-written file is locked by the AV scanner or its
/// directory entry has not settled.
#[cfg(target_os = "windows")]
fn retry_io<T>(mut f: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    let mut last = None;
    for _ in 0..5 {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) => {
                last = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    }
    Err(last.unwrap())
}
