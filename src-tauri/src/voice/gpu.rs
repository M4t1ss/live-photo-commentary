//! Where speech recognition runs: on a GPU if the user wants that, or, when
//! left to choose, if there is one with room for it. The vision model has
//! priority for the video memory: it is loaded first, and whisper takes what
//! is left (see `AppState::reinit_describer`).

use serde::Serialize;

/// What a model is expected to take of a GPU's memory: its file size times
/// this (the weights, plus working buffers that grow with them) ...
const SIZE_FACTOR: u64 = 2;
/// ... plus this much more.
const HEADROOM: u64 = 512 << 20;

/// The `voice_use_gpu` setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseGpu {
    /// Use a GPU if there is a real one with enough free memory.
    Auto,
    Cpu,
    /// Use the GPU with the most free memory, however little that is.
    Gpu,
}

impl UseGpu {
    /// From the config; a value it doesn't know is `Auto`.
    pub fn from_config(value: &str) -> Self {
        match value {
            "cpu" => Self::Cpu,
            "gpu" => Self::Gpu,
            _ => Self::Auto,
        }
    }
}

/// A GPU whisper could run on, as the machine has it right now.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuInfo {
    /// What whisper calls the device (its `gpu_device`).
    pub id: i32,
    pub name: String,
    /// Bytes of video memory free now, and in all.
    pub free: u64,
    pub total: u64,
    /// A GPU that shares the CPU's memory and is no faster than it.
    pub integrated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Placement {
    Cpu { reason: String },
    Gpu { device: i32, name: String },
}

/// How much video memory a model is expected to need.
pub fn memory_needed(model_bytes: u64) -> u64 {
    model_bytes * SIZE_FACTOR + HEADROOM
}

/// Decides where a model of `model_bytes` runs, given the GPUs there are.
pub fn place(setting: UseGpu, model_bytes: u64, gpus: &[GpuInfo]) -> Placement {
    if setting == UseGpu::Cpu {
        return Placement::Cpu { reason: "set to the CPU".to_string() };
    }
    // The automatic choice leaves integrated GPUs out.
    let usable: Vec<&GpuInfo> = gpus.iter().filter(|g| setting == UseGpu::Gpu || !g.integrated).collect();
    let Some(best) = usable.iter().max_by_key(|g| g.free) else {
        let reason = if gpus.is_empty() {
            "no GPU available to it"
        } else {
            "only an integrated GPU, which is no faster than the CPU"
        };
        return Placement::Cpu { reason: reason.to_string() };
    };
    let needed = memory_needed(model_bytes);
    if setting == UseGpu::Auto && best.free < needed {
        return Placement::Cpu {
            reason: format!(
                "not enough free memory on {} ({} MB free, about {} MB needed)",
                best.name,
                best.free >> 20,
                needed >> 20
            ),
        };
    }
    Placement::Gpu { device: best.id, name: best.name.clone() }
}

/// What the settings screen shows.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AsrStatus {
    /// `"GPU"` or `"CPU"`.
    pub device: String,
    /// The GPU's name, or why it is the CPU.
    pub detail: String,
}

impl From<&Placement> for AsrStatus {
    fn from(placement: &Placement) -> Self {
        match placement {
            Placement::Gpu { name, .. } => Self { device: "GPU".to_string(), detail: name.clone() },
            Placement::Cpu { reason } => Self { device: "CPU".to_string(), detail: reason.clone() },
        }
    }
}

/// The helper's GPUs as [`GpuInfo`]s; llama.cpp, which sees the same
/// machine, knows which of them are integrated.
pub fn gpu_infos(helper: &[whisper_protocol::Gpu], llama: &[vlm_describer::BackendDevice]) -> Vec<GpuInfo> {
    helper
        .iter()
        .map(|gpu| GpuInfo {
            id: gpu.id,
            name: gpu.name.clone(),
            free: gpu.free,
            total: gpu.total,
            integrated: llama.iter().any(|l| l.integrated && l.description == gpu.name),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1 << 20;

    fn gpu(id: i32, name: &str, free_mb: u64, integrated: bool) -> GpuInfo {
        GpuInfo { id, name: name.to_string(), free: free_mb * MB, total: 8192 * MB, integrated }
    }

    fn base() -> u64 {
        60 * MB
    }

    #[test]
    fn the_setting_comes_from_the_config_and_unknown_values_are_automatic() {
        assert_eq!(UseGpu::from_config("cpu"), UseGpu::Cpu);
        assert_eq!(UseGpu::from_config("gpu"), UseGpu::Gpu);
        assert_eq!(UseGpu::from_config("auto"), UseGpu::Auto);
        assert_eq!(UseGpu::from_config("??"), UseGpu::Auto);
    }

    #[test]
    fn a_model_is_expected_to_take_twice_its_size_and_half_a_gigabyte() {
        assert_eq!(memory_needed(60 * MB), 632 * MB);
        assert_eq!(memory_needed(574 * MB), 1660 * MB);
    }

    #[test]
    fn set_to_the_cpu_it_stays_there_whatever_the_gpus() {
        let gpus = [gpu(0, "RTX", 6000, false)];
        assert_eq!(place(UseGpu::Cpu, base(), &gpus), Placement::Cpu { reason: "set to the CPU".to_string() });
    }

    #[test]
    fn automatically_it_takes_a_gpu_with_room() {
        let gpus = [gpu(0, "RTX 3060 Ti", 3000, false)];
        assert_eq!(place(UseGpu::Auto, base(), &gpus), Placement::Gpu { device: 0, name: "RTX 3060 Ti".to_string() });
    }

    #[test]
    fn automatically_it_stays_on_the_cpu_when_the_gpu_is_nearly_full() {
        // The vision model and a game took the memory.
        let gpus = [gpu(0, "RTX 3060 Ti", 400, false)];
        match place(UseGpu::Auto, base(), &gpus) {
            Placement::Cpu { reason } => {
                assert!(reason.contains("not enough free memory") && reason.contains("RTX 3060 Ti"), "{reason}");
                assert!(reason.contains("400 MB free") && reason.contains("632 MB needed"), "{reason}");
            }
            other => panic!("expected the CPU, got {other:?}"),
        }
    }

    #[test]
    fn the_room_needed_grows_with_the_model() {
        let gpus = [gpu(0, "RTX", 1000, false)];
        assert!(matches!(place(UseGpu::Auto, 60 * MB, &gpus), Placement::Gpu { .. }));
        assert!(matches!(place(UseGpu::Auto, 574 * MB, &gpus), Placement::Cpu { .. }), "the large model needs 1660 MB");
    }

    #[test]
    fn automatically_an_integrated_gpu_is_not_used_but_a_discrete_one_next_to_it_is() {
        let integrated = [gpu(0, "Intel UHD", 8000, true)];
        match place(UseGpu::Auto, base(), &integrated) {
            Placement::Cpu { reason } => assert!(reason.contains("integrated"), "{reason}"),
            other => panic!("expected the CPU, got {other:?}"),
        }
        let both = [gpu(0, "Intel UHD", 8000, true), gpu(1, "RTX", 3000, false)];
        assert_eq!(place(UseGpu::Auto, base(), &both), Placement::Gpu { device: 1, name: "RTX".to_string() });
    }

    #[test]
    fn it_takes_the_gpu_with_the_most_free_memory() {
        let gpus = [gpu(0, "small card", 1500, false), gpu(1, "big card", 5000, false)];
        assert_eq!(place(UseGpu::Auto, base(), &gpus), Placement::Gpu { device: 1, name: "big card".to_string() });
    }

    #[test]
    fn without_any_gpu_it_is_the_cpu() {
        for setting in [UseGpu::Auto, UseGpu::Gpu] {
            match place(setting, base(), &[]) {
                Placement::Cpu { reason } => assert!(reason.contains("no GPU"), "{reason}"),
                other => panic!("expected the CPU, got {other:?}"),
            }
        }
    }

    #[test]
    fn forced_to_the_gpu_it_tries_even_without_room_or_with_only_an_integrated_one() {
        let nearly_full = [gpu(0, "RTX", 100, false)];
        assert_eq!(place(UseGpu::Gpu, base(), &nearly_full), Placement::Gpu { device: 0, name: "RTX".to_string() });
        let integrated = [gpu(0, "Intel UHD", 100, true)];
        assert!(matches!(place(UseGpu::Gpu, base(), &integrated), Placement::Gpu { .. }));
    }

    #[test]
    fn the_status_says_where_and_why() {
        let gpu = Placement::Gpu { device: 0, name: "RTX 3060 Ti".to_string() };
        assert_eq!(AsrStatus::from(&gpu), AsrStatus { device: "GPU".to_string(), detail: "RTX 3060 Ti".to_string() });
        let cpu = Placement::Cpu { reason: "set to the CPU".to_string() };
        assert_eq!(AsrStatus::from(&cpu), AsrStatus { device: "CPU".to_string(), detail: "set to the CPU".to_string() });
    }

    #[test]
    fn the_helpers_gpus_are_integrated_when_llama_cpp_says_so() {
        let llama = |description: &str, integrated: bool| vlm_describer::BackendDevice {
            index: 0,
            backend: "Vulkan".to_string(),
            description: description.to_string(),
            memory_total: 0,
            gpu: true,
            integrated,
        };
        let helper = [
            whisper_protocol::Gpu { id: 0, name: "Intel UHD".to_string(), free: 1 << 30, total: 2 << 30 },
            whisper_protocol::Gpu { id: 1, name: "RTX 3060 Ti".to_string(), free: 6 << 30, total: 8 << 30 },
        ];
        let infos = gpu_infos(&helper, &[llama("Intel UHD", true), llama("RTX 3060 Ti", false)]);
        assert_eq!(infos.iter().map(|g| (g.id, g.integrated)).collect::<Vec<_>>(), [(0, true), (1, false)]);
        assert_eq!((infos[1].free, infos[1].total), (6 << 30, 8 << 30));
        // A GPU llama.cpp does not know is taken to be a discrete one.
        assert!(!gpu_infos(&helper, &[])[0].integrated);
    }
}
