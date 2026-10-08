//! The optional CUDA backend for llama.cpp, downloaded from Settings.
//!
//! The installer ships Vulkan, which is nearly as fast. CUDA is an archive of
//! the CUDA backend library (built by CI for exactly this app version, so it
//! matches the `ggml` libraries in the installer) plus NVIDIA's cuBLAS
//! libraries, about 600 MB. It lives in `<app data>/backends/cuda/<version>/`
//! and is loaded at startup when it is there. Every function takes the
//! `backends/cuda` folder (`root`), so the tests can use a temporary one.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};

/// About how big the download is, for the Settings text.
pub const DOWNLOAD_SIZE_MB: u64 = 600;

/// The archive's checksum, set when the app is built (`CUDA_BACKEND_SHA256`):
/// CI builds the archive first and then the installer. Without it (a
/// development build) nothing can be downloaded.
pub const SHA256: Option<&str> = option_env!("CUDA_BACKEND_SHA256");

/// Set while a download runs, so Settings can't start a second one.
static DOWNLOADING: AtomicBool = AtomicBool::new(false);

/// The name of the folder inside `root` for this app version, so that a
/// backend built for another version is never loaded.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// What the archive is called on this platform, if CUDA is offered on it.
fn archive_name() -> Option<&'static str> {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some("cuda-backend-windows-x64.zip")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("cuda-backend-linux-x64.zip")
    } else {
        None
    }
}

/// Where the archive for this version is: a release asset of the matching
/// tag, unless the build says otherwise (`CUDA_BACKEND_URL`, for tests).
pub fn download_url() -> Option<String> {
    if let Some(url) = option_env!("CUDA_BACKEND_URL") {
        return Some(url.to_string());
    }
    Some(format!(
        "https://github.com/M4t1ss/live-photo-commentary/releases/download/v{VERSION}/{}",
        archive_name()?
    ))
}

/// Whether this build can offer to download CUDA at all.
pub fn available() -> bool {
    SHA256.is_some() && download_url().is_some()
}

fn version_dir(root: &Path) -> PathBuf {
    root.join(VERSION)
}

/// A file whose presence means "delete the installed backend at the next
/// start": its library is loaded now, so Windows won't delete it.
fn removal_marker(root: &Path) -> PathBuf {
    root.join("remove-on-start")
}

/// Whether "Remove CUDA" was pressed and will finish at the next start.
pub fn removal_pending(root: &Path) -> bool {
    removal_marker(root).exists()
}

pub fn installed(root: &Path) -> bool {
    version_dir(root).is_dir() && !removal_pending(root)
}

/// Finishes a removal that had to wait for a restart, and deletes backends
/// installed by other app versions or left half-unpacked. Call at startup,
/// before loading.
pub fn clean_up(root: &Path) {
    let remove_all = removal_pending(root);
    let current = version_dir(root);
    let Ok(entries) = std::fs::read_dir(root) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && (remove_all || path != current) {
            if let Err(e) = std::fs::remove_dir_all(&path) {
                log::warn!("cannot delete {}: {e}", path.display());
            }
        }
    }
    if remove_all {
        let _ = std::fs::remove_file(removal_marker(root));
    }
}

/// Loads the installed backend, if there is one (startup). Do it before
/// loading the installer's backends: the first GPU llama.cpp knows is the one
/// its image encoder uses, and that should be the CUDA one.
pub fn load_installed(root: &Path) {
    if installed(root) {
        load(&version_dir(root));
    }
}

/// Loads the CUDA backend in `dir`. NVIDIA's libraries are next to it, but
/// Windows looks for the libraries of a library in the application's folder
/// and the system's, not in its own, so `dir` is added to the search path.
pub(crate) fn load(dir: &Path) {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        // SAFETY: `wide` is a null-terminated UTF-16 string that outlives the call.
        unsafe { windows_sys::Win32::System::LibraryLoader::SetDllDirectoryW(wide.as_ptr()) };
    }
    vlm_describer::load_backends(Some(dir));
}

/// Whether llama.cpp has a CUDA device now.
pub fn loaded() -> bool {
    vlm_describer::backend_devices().iter().any(|device| device.backend == "CUDA")
}

/// The first NVIDIA graphics card, whichever backend reports it.
pub fn nvidia_gpu() -> Option<String> {
    vlm_describer::backend_devices()
        .into_iter()
        .find(|device| device.gpu && device.description.to_ascii_lowercase().contains("nvidia"))
        .map(|device| device.description)
}

/// Downloads, checks and unpacks the backend, then loads it. `progress` gets
/// the bytes done and the total, at most once per percent.
pub async fn install(
    root: &Path,
    url: &str,
    expected_sha256: &str,
    progress: impl Fn(u64, u64),
) -> Result<(), String> {
    if DOWNLOADING.swap(true, Ordering::SeqCst) {
        return Err("the CUDA backend is already downloading".to_string());
    }
    let result = download_and_unpack(root, url, expected_sha256, progress).await;
    DOWNLOADING.store(false, Ordering::SeqCst);
    let dir = result?;
    let _ = std::fs::remove_file(removal_marker(root));
    load(&dir);
    if loaded() {
        Ok(())
    } else {
        Err("the CUDA backend was installed but did not load; the NVIDIA driver may be too old (CUDA 13 needs 580 or newer)".to_string())
    }
}

async fn download_and_unpack(
    root: &Path,
    url: &str,
    expected_sha256: &str,
    progress: impl Fn(u64, u64),
) -> Result<PathBuf, String> {
    use tokio::io::AsyncWriteExt;

    std::fs::create_dir_all(root).map_err(|e| format!("cannot create {}: {e}", root.display()))?;
    let archive = root.join("download.zip");

    let mut response = reqwest::get(url)
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("cannot download the CUDA backend: {e}"))?;
    let total = response.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(&archive).await.map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let (mut done, mut last_percent) = (0u64, u64::MAX);
    while let Some(chunk) = response.chunk().await.map_err(|e| format!("download interrupted: {e}"))? {
        hasher.update(&chunk);
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        done += chunk.len() as u64;
        let percent = if total > 0 { done * 100 / total } else { 0 };
        if percent != last_percent {
            last_percent = percent;
            progress(done, total);
        }
    }
    file.flush().await.map_err(|e| e.to_string())?;
    drop(file);

    let actual: String = hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect();
    if !actual.eq_ignore_ascii_case(expected_sha256) {
        let _ = std::fs::remove_file(&archive);
        return Err("the downloaded CUDA backend is corrupt (checksum mismatch)".to_string());
    }

    // Unpack beside the target and rename, so an interrupted unpacking never
    // looks like an installed backend (`clean_up` deletes what is left).
    let target = version_dir(root);
    let unpacking = root.join("unpacking");
    let (archive_path, unpacking_path, target_path) = (archive.clone(), unpacking, target.clone());
    tokio::task::spawn_blocking(move || -> Result<(), String> {
        let _ = std::fs::remove_dir_all(&unpacking_path);
        let file = std::fs::File::open(&archive_path).map_err(|e| e.to_string())?;
        zip::ZipArchive::new(file)
            .and_then(|mut zip| zip.extract(&unpacking_path))
            .map_err(|e| format!("cannot unpack the CUDA backend: {e}"))?;
        let _ = std::fs::remove_dir_all(&target_path);
        std::fs::rename(&unpacking_path, &target_path).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())??;
    let _ = std::fs::remove_file(&archive);
    Ok(target)
}

/// Deletes the installed backend. If its library is loaded, Windows refuses,
/// and the files stay until the next start (see [`removal_marker`]).
pub fn remove(root: &Path) -> Result<(), String> {
    let dir = version_dir(root);
    if !dir.is_dir() {
        return Ok(());
    }
    if std::fs::remove_dir_all(&dir).is_err() {
        std::fs::write(removal_marker(root), b"").map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A temporary folder that is deleted afterwards.
    struct Temp(PathBuf);

    impl Temp {
        fn new(name: &str) -> Temp {
            let dir = std::env::temp_dir().join(format!("lpc-cuda-test-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Temp(dir)
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn zip_with(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(&mut bytes);
        for (name, contents) in files {
            zip.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            zip.write_all(contents).unwrap();
        }
        zip.finish().unwrap();
        bytes.into_inner()
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
    }

    async fn serve(archive: Vec<u8>) -> wiremock::MockServer {
        use wiremock::matchers::{method, path};
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(method("GET"))
            .and(path("/cuda.zip"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_bytes(archive))
            .mount(&server)
            .await;
        server
    }

    // `install` ends by loading the backend, which these archives don't hold,
    // so the tests call `download_and_unpack`, the part before that.
    #[tokio::test]
    async fn downloads_checks_and_unpacks_the_archive() {
        let temp = Temp::new("unpack");
        let archive = zip_with(&[("ggml-cuda.dll", b"backend"), ("cublas64_13.dll", b"cublas")]);
        let sha = sha256_hex(&archive);
        let server = serve(archive).await;

        let progress = std::sync::Mutex::new(Vec::new());
        let dir = download_and_unpack(&temp.0, &format!("{}/cuda.zip", server.uri()), &sha, |done, total| {
            progress.lock().unwrap().push((done, total));
        })
        .await
        .unwrap();

        assert_eq!(dir, version_dir(&temp.0));
        assert_eq!(std::fs::read(dir.join("ggml-cuda.dll")).unwrap(), b"backend");
        assert_eq!(std::fs::read(dir.join("cublas64_13.dll")).unwrap(), b"cublas");
        assert!(!temp.0.join("download.zip").exists(), "the archive is deleted");
        assert!(installed(&temp.0));
        let progress = progress.lock().unwrap();
        assert_eq!(progress.last().map(|&(done, total)| done == total), Some(true));
    }

    #[tokio::test]
    async fn refuses_an_archive_with_the_wrong_checksum() {
        let temp = Temp::new("checksum");
        let server = serve(zip_with(&[("ggml-cuda.dll", b"backend")])).await;
        let error = download_and_unpack(&temp.0, &format!("{}/cuda.zip", server.uri()), &"0".repeat(64), |_, _| {})
            .await
            .unwrap_err();
        assert!(error.contains("checksum"), "{error}");
        assert!(!installed(&temp.0));
        assert!(!temp.0.join("download.zip").exists(), "a bad archive is deleted");
    }

    #[tokio::test]
    async fn reports_a_missing_file() {
        let temp = Temp::new("missing");
        let server = serve(Vec::new()).await;
        let error = download_and_unpack(&temp.0, &format!("{}/other.zip", server.uri()), "abc", |_, _| {})
            .await
            .unwrap_err();
        assert!(error.contains("cannot download"), "{error}");
    }

    #[test]
    fn remove_deletes_the_folder() {
        let temp = Temp::new("remove");
        std::fs::create_dir_all(version_dir(&temp.0)).unwrap();
        std::fs::write(version_dir(&temp.0).join("ggml-cuda.dll"), b"x").unwrap();
        assert!(installed(&temp.0));
        remove(&temp.0).unwrap();
        assert!(!installed(&temp.0));
        assert!(!removal_pending(&temp.0));
        remove(&temp.0).unwrap();
    }

    #[test]
    fn a_pending_removal_hides_the_backend_and_clean_up_finishes_it() {
        let temp = Temp::new("pending");
        std::fs::create_dir_all(version_dir(&temp.0)).unwrap();
        std::fs::write(removal_marker(&temp.0), b"").unwrap();
        assert!(!installed(&temp.0), "it must not be loaded again");
        assert!(removal_pending(&temp.0));
        clean_up(&temp.0);
        assert!(!version_dir(&temp.0).exists());
        assert!(!removal_pending(&temp.0));
    }

    #[test]
    fn clean_up_deletes_other_versions_and_half_unpacked_downloads_only() {
        let temp = Temp::new("others");
        for name in [VERSION, "0.0.1", "unpacking"] {
            std::fs::create_dir_all(temp.0.join(name)).unwrap();
        }
        clean_up(&temp.0);
        assert!(version_dir(&temp.0).is_dir());
        assert!(!temp.0.join("0.0.1").exists());
        assert!(!temp.0.join("unpacking").exists());
    }

    /// Installs the real archive made by `cargo xtask package-cuda-backend`
    /// (`CUDA_BACKEND_ZIP` is its path) from a local server, as the Settings
    /// button does, and checks that the backend loads, then that removing it
    /// while it is loaded waits for the next start. Needs an NVIDIA GPU and
    /// the base libraries of an app build with `dynamic-backends`:
    /// `CUDA_BACKEND_ZIP=... cargo test -p app --lib --features dynamic-backends,vulkan -- --ignored --nocapture real_cuda`
    #[tokio::test]
    #[ignore]
    async fn real_cuda_backend_installs_loads_and_waits_to_be_removed() {
        let Some(zip_path) = std::env::var_os("CUDA_BACKEND_ZIP") else {
            eprintln!("skipping: CUDA_BACKEND_ZIP not set");
            return;
        };
        let archive = std::fs::read(zip_path).expect("read the archive");
        let sha = sha256_hex(&archive);
        let server = serve(archive).await;
        let temp = Temp::new("real");

        vlm_describer::load_backends(None);
        assert!(!loaded(), "CUDA must not be loaded yet");
        install(&temp.0, &format!("{}/cuda.zip", server.uri()), &sha, |done, total| {
            if done == total {
                println!("downloaded {done} bytes");
            }
        })
        .await
        .expect("install");
        assert!(loaded());
        println!("NVIDIA GPU: {:?}", nvidia_gpu());
        assert!(installed(&temp.0));

        remove(&temp.0).expect("remove");
        assert!(removal_pending(&temp.0), "Windows keeps the loaded DLL, so the removal waits");
        assert!(!installed(&temp.0));
        clean_up(&temp.0); // the next start: still locked in this process, so only the marker's folder is tried
        println!("after clean_up: folder exists {}", version_dir(&temp.0).exists());
    }

    #[test]
    fn the_download_url_names_this_version() {
        if let Some(url) = download_url() {
            assert!(option_env!("CUDA_BACKEND_URL").is_some() || url.contains(&format!("v{VERSION}")), "{url}");
        }
    }
}
