// ONNX Runtime's prebuilt Windows binaries (from the `ort` crate) always
// include DirectML, so programs import DirectML.dll and won't start without
// it. This crate only uses the CPU, so delay-load the DLL: Windows then only
// looks for it if DirectML is actually used, which never happens.
//
// Link arguments from a build script only apply to this package's own
// programs (examples and tests). Applications using this crate need the same
// two arguments in their own build script.
fn main() {
    let windows_msvc = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    if windows_msvc {
        println!("cargo:rustc-link-arg=/DELAYLOAD:DirectML.dll");
        println!("cargo:rustc-link-arg=delayimp.lib");
    }
}
