//! Which compute backend llama.cpp runs local models on: CPU, or a GPU
//! backend (Vulkan, Metal, CUDA) that was built in or loaded at run time.

use serde::Serialize;
use tauri::Manager;
use vlm_describer::BackendDevice;

/// The folder of the optional, downloaded CUDA backend.
pub fn cuda_root(app: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    Ok(app.path().app_data_dir().map_err(|e| e.to_string())?.join("backends").join("cuda"))
}

/// Loads llama.cpp's backend libraries. Call once at startup, before any
/// model is loaded. The installed CUDA backend (if any) comes first, then
/// the installer's: in release builds they are in the bundled `backends`
/// folder; in debug builds, in the folder they were built into.
pub fn init(app: &tauri::AppHandle) {
    if let Ok(root) = cuda_root(app) {
        crate::cuda_backend::clean_up(&root);
        crate::cuda_backend::load_installed(&root);
    }
    let bundled = app.path().resource_dir().ok().map(|dir| dir.join("backends")).filter(|dir| dir.is_dir());
    vlm_describer::load_backends(bundled.as_deref());
    for device in vlm_describer::backend_devices() {
        log::info!("compute device: {} {} (GPU: {})", device.backend, device.description, device.gpu);
    }
}

/// How many layers of a local model to put on the GPU: all of them if a GPU
/// is available to llama.cpp, else none.
pub fn gpu_layers() -> u32 {
    if vlm_describer::backend_devices().iter().any(|device| device.gpu) {
        999
    } else {
        0
    }
}

/// What the Settings "GPU acceleration" section shows.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GpuStatus {
    /// `"CPU"`, or the GPU backend in use: `"Vulkan"`, `"CUDA"`, `"Metal"`.
    pub backend: String,
    /// The GPU in use, if any.
    pub gpu: Option<String>,
    /// An NVIDIA graphics card, if the machine has one.
    pub nvidia_gpu: Option<String>,
    /// Whether this app can download the CUDA backend.
    pub cuda_downloadable: bool,
    pub cuda_size_mb: u64,
    pub cuda_installed: bool,
    /// "Remove CUDA" was pressed, but the files go at the next start.
    pub cuda_removal_pending: bool,
}

/// The GPU local models run on (`vlm_describer::preferred_gpu`), else the CPU.
fn gpu_for(devices: &[BackendDevice]) -> (String, Option<String>) {
    match vlm_describer::preferred_gpu(devices) {
        Some(device) => (device.backend.clone(), Some(device.description.clone())),
        None => ("CPU".to_string(), None),
    }
}

pub fn status(app: &tauri::AppHandle) -> Result<GpuStatus, String> {
    let root = cuda_root(app)?;
    let (backend, gpu) = gpu_for(&vlm_describer::backend_devices());
    Ok(GpuStatus {
        backend,
        gpu,
        nvidia_gpu: crate::cuda_backend::nvidia_gpu(),
        cuda_downloadable: crate::cuda_backend::available(),
        cuda_size_mb: crate::cuda_backend::DOWNLOAD_SIZE_MB,
        cuda_installed: crate::cuda_backend::installed(&root),
        cuda_removal_pending: crate::cuda_backend::removal_pending(&root),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(backend: &str, description: &str, gpu: bool) -> BackendDevice {
        BackendDevice {
            index: 0,
            backend: backend.to_string(),
            description: description.to_string(),
            memory_total: 0,
            gpu,
            integrated: false,
        }
    }

    #[test]
    fn the_preferred_gpu_decides_the_status() {
        let devices = [device("CPU", "13th Gen Intel", false), device("Vulkan", "RTX 3060 Ti", true), device("CUDA", "RTX 3060 Ti", true)];
        assert_eq!(gpu_for(&devices), ("CUDA".to_string(), Some("RTX 3060 Ti".to_string())));
    }

    #[test]
    fn without_a_gpu_it_is_the_cpu() {
        assert_eq!(gpu_for(&[device("CPU", "x", false)]), ("CPU".to_string(), None));
        assert_eq!(gpu_for(&[]), ("CPU".to_string(), None));
    }
}
