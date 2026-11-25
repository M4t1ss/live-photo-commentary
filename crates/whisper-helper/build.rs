fn main() {
    // With Vulkan, the executable would import the Vulkan loader and not start on
    // a machine without it. Delay-load it instead; `gpus` checks that it is
    // there before asking for the GPUs (see src/whisper.rs).
    let windows_msvc = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    if windows_msvc && std::env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        println!("cargo:rustc-link-arg=/DELAYLOAD:vulkan-1.dll");
        println!("cargo:rustc-link-arg=delayimp.lib");
    }
}
