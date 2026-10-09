//! Which compute backends llama.cpp can use: loading them at run time and
//! listing the devices they provide.

use std::path::Path;

use llama_cpp_2::{LlamaBackendDeviceType, list_llama_ggml_backend_devices};

/// A device llama.cpp can run on.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendDevice {
    /// llama.cpp's number for the device.
    pub index: usize,
    /// The backend that provides it: `"CPU"`, `"Vulkan"`, `"CUDA"`, `"Metal"`.
    pub backend: String,
    /// For example `"NVIDIA GeForce RTX 3060 Ti"`.
    pub description: String,
    /// Bytes of memory, as far as the driver says.
    pub memory_total: u64,
    /// Whether this is a graphics card (including an integrated one) rather
    /// than the CPU.
    pub gpu: bool,
    /// Whether it is a GPU that shares the CPU's memory.
    pub integrated: bool,
}

/// Loads the backend libraries in `dir` (`ggml-cpu-*`, `ggml-vulkan`,
/// `ggml-cuda`, ...), or from the folder they were built into if `dir` is
/// `None` (for development). Call it before loading a model, and again for
/// another folder to add its backends.
///
/// With the `dynamic-backends` feature, llama.cpp is built with its backends
/// as separate libraries, and none is available until they are loaded. Without
/// it, the backends are built in and this does nothing.
pub fn load_backends(dir: Option<&Path>) {
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
    #[cfg(feature = "dynamic-backends")]
    match dir {
        Some(dir) => llama_cpp_2::llama_backend::load_backends_from_path(dir),
        None => llama_cpp_2::llama_backend::load_backends(),
    }
    #[cfg(not(feature = "dynamic-backends"))]
    log::debug!("backends are built in; not loading {dir:?}");
}

/// The devices of the backends loaded so far, in llama.cpp's order.
pub fn backend_devices() -> Vec<BackendDevice> {
    list_llama_ggml_backend_devices()
        .into_iter()
        .map(|device| BackendDevice {
            index: device.index,
            backend: device.backend,
            description: device.description,
            memory_total: device.memory_total as u64,
            gpu: matches!(
                device.device_type,
                LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu
            ),
            integrated: device.device_type == LlamaBackendDeviceType::IntegratedGpu,
        })
        .collect()
}

/// The GPU to run a model on, among `devices`. The same card can show up once
/// per backend (CUDA and Vulkan, say), and llama.cpp would spread a model over
/// all of them, so the model is pinned to one. CUDA is preferred, then the
/// discrete GPU with the most memory, then an integrated one.
pub fn preferred_gpu(devices: &[BackendDevice]) -> Option<&BackendDevice> {
    devices
        .iter()
        .filter(|device| device.gpu)
        .max_by_key(|device| (device.backend == "CUDA", !device.integrated, device.memory_total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(index: usize, backend: &str, gpu: bool, integrated: bool, gigabytes: u64) -> BackendDevice {
        BackendDevice {
            index,
            backend: backend.to_string(),
            description: format!("device {index}"),
            memory_total: gigabytes << 30,
            gpu,
            integrated,
        }
    }

    #[test]
    fn cuda_is_preferred_over_the_same_card_on_vulkan() {
        let devices = [
            device(0, "CPU", false, false, 32),
            device(1, "Vulkan", true, false, 8),
            device(2, "CUDA", true, false, 8),
        ];
        assert_eq!(preferred_gpu(&devices).unwrap().index, 2);
    }

    #[test]
    fn a_discrete_gpu_beats_an_integrated_one_and_more_memory_wins() {
        let devices = [
            device(0, "Vulkan", true, true, 16),
            device(1, "Vulkan", true, false, 6),
            device(2, "Vulkan", true, false, 12),
        ];
        assert_eq!(preferred_gpu(&devices).unwrap().index, 2);
        assert_eq!(preferred_gpu(&devices[..1]).unwrap().index, 0);
    }

    #[test]
    fn without_a_gpu_there_is_none() {
        assert!(preferred_gpu(&[device(0, "CPU", false, false, 32)]).is_none());
        assert!(preferred_gpu(&[]).is_none());
    }
}
