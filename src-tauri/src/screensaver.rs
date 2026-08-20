// ── Screensaver / display-sleep inhibit ──────────────────────────────────────
//
// Prevents screensaver and display sleep while the describer loop runs.
// Call inhibit_screensaver on start, allow_screensaver on stop.
//
// Windows  – SetThreadExecutionState (Win32, stateless from our side)
// macOS    – IOPMAssertionCreateWithName / IOPMAssertionRelease (IOKit)
// Linux    – org.freedesktop.ScreenSaver D-Bus Inhibit / UnInhibit cookie

use std::sync::Mutex;
use tauri::State;

// macOS and Linux need to remember a handle/cookie to release the inhibit.
// Windows SetThreadExecutionState is stateless — we just call it each time.
pub struct ScreensaverState {
    #[allow(dead_code)] // unused on Windows where SetThreadExecutionState is stateless
    id: Mutex<Option<u32>>,
}

impl Default for ScreensaverState {
    fn default() -> Self {
        Self { id: Mutex::new(None) }
    }
}

#[tauri::command]
pub fn inhibit_screensaver(_state: State<ScreensaverState>) {
    #[cfg(windows)]
    windows::inhibit();

    #[cfg(not(windows))]
    {
        let mut guard = _state.id.lock().unwrap();
        if guard.is_some() {
            return;
        }
        #[cfg(target_os = "macos")]
        { *guard = macos::inhibit(); }
        #[cfg(not(target_os = "macos"))]
        { *guard = linux::inhibit(); }
    }
}

#[tauri::command]
pub fn allow_screensaver(_state: State<ScreensaverState>) {
    #[cfg(windows)]
    windows::allow();

    #[cfg(not(windows))]
    if let Some(id) = _state.id.lock().unwrap().take() {
        #[cfg(target_os = "macos")]
        macos::allow(id);
        #[cfg(not(target_os = "macos"))]
        linux::allow(id);
    }
}

// ── Windows ───────────────────────────────────────────────────────────────────

#[cfg(windows)]
mod windows {
    use windows_sys::Win32::System::Power::{
        SetThreadExecutionState, ES_CONTINUOUS, ES_DISPLAY_REQUIRED, ES_SYSTEM_REQUIRED,
    };

    pub fn inhibit() {
        unsafe { SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED); }
    }

    pub fn allow() {
        unsafe { SetThreadExecutionState(ES_CONTINUOUS); }
    }
}

// ── macOS ─────────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::{c_char, c_void};

    type IOReturn = i32;
    type IOPMAssertionID = u32;
    type IOPMAssertionLevel = u32;

    const LEVEL_ON: IOPMAssertionLevel = 255;
    const CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

    #[link(name = "IOKit", kind = "framework")]
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn IOPMAssertionCreateWithName(
            assertion_type: *const c_void,
            assertion_level: IOPMAssertionLevel,
            assertion_name: *const c_void,
            assertion_id: *mut IOPMAssertionID,
        ) -> IOReturn;
        fn IOPMAssertionRelease(assertion_id: IOPMAssertionID) -> IOReturn;
        fn CFStringCreateWithCString(
            alloc: *const c_void,
            c_str: *const c_char,
            encoding: u32,
        ) -> *const c_void;
        fn CFRelease(cf: *const c_void);
    }

    fn cf_str(s: &[u8]) -> *const c_void {
        unsafe {
            CFStringCreateWithCString(
                std::ptr::null(),
                s.as_ptr() as *const c_char,
                CF_STRING_ENCODING_UTF8,
            )
        }
    }

    pub fn inhibit() -> Option<u32> {
        unsafe {
            let ty = cf_str(b"PreventUserIdleDisplaySleep\0");
            let name = cf_str(b"Describer loop running\0");
            let mut id: IOPMAssertionID = 0;
            let ret = IOPMAssertionCreateWithName(ty, LEVEL_ON, name, &mut id);
            CFRelease(ty);
            CFRelease(name);
            if ret == 0 { Some(id) } else { None }
        }
    }

    pub fn allow(id: u32) {
        unsafe { IOPMAssertionRelease(id); }
    }
}

// ── Linux ─────────────────────────────────────────────────────────────────────

#[cfg(all(unix, not(target_os = "macos")))]
mod linux {
    use std::time::Duration;

    const DEST: &str = "org.freedesktop.ScreenSaver";
    const PATH: &str = "/ScreenSaver";
    const IFACE: &str = "org.freedesktop.ScreenSaver";

    pub fn inhibit() -> Option<u32> {
        let conn = dbus::blocking::Connection::new_session().ok()?;
        let proxy = conn.with_proxy(DEST, PATH, Duration::from_secs(5));
        let (cookie,): (u32,) = proxy
            .method_call(IFACE, "Inhibit", ("Live Photo Commentary", "Describer loop running"))
            .ok()?;
        Some(cookie)
    }

    pub fn allow(cookie: u32) {
        let Ok(conn) = dbus::blocking::Connection::new_session() else { return };
        let proxy = conn.with_proxy(DEST, PATH, Duration::from_secs(5));
        let _: Result<(), _> = proxy.method_call(IFACE, "UnInhibit", (cookie,));
    }
}
