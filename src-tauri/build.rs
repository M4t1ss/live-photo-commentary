fn main() {
  // kokoro-timestamped's ONNX Runtime binaries for Windows always import
  // DirectML.dll, which is never used (CPU only). Delay-load it so the app
  // starts without it; see kokoro-timestamped's README.
  let windows_msvc = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
    && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
  if windows_msvc {
    println!("cargo:rustc-link-arg=/DELAYLOAD:DirectML.dll");
    println!("cargo:rustc-link-arg=delayimp.lib");
  }
  tauri_build::build()
}
