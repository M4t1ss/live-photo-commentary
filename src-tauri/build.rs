use std::path::{Path, PathBuf};

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
  stage_llama_libraries();
  // With `dynamic-backends`, libllama and libggml are shared libraries that the
  // bundle puts in Tauri's resource folder, `/usr/lib/<product name>` in Linux
  // packages and AppImages, next to `/usr/bin/<executable>`. Without an rpath
  // the loader doesn't look there. (Untested: written without a Linux machine.)
  let dynamic_backends = std::env::var_os("CARGO_FEATURE_DYNAMIC_BACKENDS").is_some();
  if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") && dynamic_backends {
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib/Interactive Live Commentary Assistant");
  }
  // macOS: the same libraries are in `Contents/Resources`, next to `Contents/MacOS`
  // where the executable is; their install names are `@rpath/...`. (Untested too.)
  if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") && dynamic_backends {
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Resources");
  }
  tauri_build::build()
}

/// Collects llama.cpp's shared libraries and backend modules in
/// `resources/llama/` (and `resources/llama/backends/`), which
/// `tauri.conf.json` bundles next to the executable (and `tauri dev` copies
/// there). Only with the `dynamic-backends` feature; otherwise llama.cpp is
/// linked in and the folder holds just a placeholder, because Tauri insists
/// that the resource pattern matches something.
///
/// A build with the `cuda` feature is the one that makes the downloadable CUDA
/// backend (`cargo xtask package-cuda-backend`), not the app: it stages into
/// `resources/llama-cuda/` instead, which isn't bundled.
fn stage_llama_libraries() {
  // Files are only written when they differ: `tauri dev` watches this folder
  // and rebuilds whenever it changes, so rewriting it on every build would
  // make it rebuild forever.
  let folder = if std::env::var_os("CARGO_FEATURE_CUDA").is_some() { "resources/llama-cuda" } else { "resources/llama" };
  let staging = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(folder);
  std::fs::create_dir_all(staging.join("backends")).unwrap();
  write_if_different(&staging.join("README.txt"), b"llama.cpp libraries, filled in by build.rs\n");
  if std::env::var_os("CARGO_FEATURE_DYNAMIC_BACKENDS").is_none() {
    return;
  }

  // `lib/cmake` (or `lib64/cmake`) inside the CMake install of llama.cpp.
  let cmake_dir = PathBuf::from(std::env::var("DEP_LLAMA_GGML_CMAKE_DIR").expect("llama-cpp-sys-2 did not say where ggml was installed"));
  let install = cmake_dir.parent().and_then(Path::parent).expect("ggml cmake dir is nested");
  // Windows puts DLLs in `bin`, the others in `lib`/`lib64`.
  for dir in ["bin", "lib", "lib64"] {
    copy_shared_libraries(&install.join(dir), &staging);
  }
  let backends = PathBuf::from(std::env::var("DEP_LLAMA_BACKENDS_DIR").expect("llama-cpp-sys-2 did not say where the backends are"));
  copy_shared_libraries(&backends, &staging.join("backends"));
}

fn copy_shared_libraries(from: &Path, to: &Path) {
  let Ok(entries) = std::fs::read_dir(from) else { return };
  for entry in entries.flatten() {
    let name = entry.file_name().to_string_lossy().into_owned();
    let shared = name.ends_with(".dll") || name.contains(".so") || name.ends_with(".dylib");
    if shared && entry.path().is_file() {
      let target = to.join(&name);
      let same = match (std::fs::metadata(entry.path()), std::fs::metadata(&target)) {
        (Ok(a), Ok(b)) => a.len() == b.len() && a.modified().ok() == b.modified().ok(),
        _ => false,
      };
      if !same {
        // `copy` keeps the modification time on Windows but not everywhere;
        // setting it explicitly makes the comparison above work on all.
        std::fs::copy(entry.path(), &target).unwrap();
        if let Ok(modified) = std::fs::metadata(entry.path()).and_then(|m| m.modified()) {
          let _ = std::fs::File::options().write(true).open(&target).and_then(|f| f.set_modified(modified));
        }
      }
    }
  }
}

fn write_if_different(path: &Path, contents: &[u8]) {
  if std::fs::read(path).ok().as_deref() != Some(contents) {
    std::fs::write(path, contents).unwrap();
  }
}
