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
