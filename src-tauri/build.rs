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
fn stage_llama_libraries() {
  // Files are only written when they differ: `tauri dev` watches this folder
  // and rebuilds whenever it changes, so rewriting it on every build would
  // make it rebuild forever.
  let staging = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("resources/llama");
  std::fs::create_dir_all(staging.join("backends")).unwrap();
  write_if_different(&staging.join("README.txt"), b"llama.cpp libraries, filled in by build.rs\n");
  if std::env::var_os("CARGO_FEATURE_DYNAMIC_BACKENDS").is_none() {
    return;
  }

  // `lib/cmake` (or `lib64/cmake`) inside the CMake install of llama.cpp.
  let cmake_dir = PathBuf::from(std::env::var("DEP_LLAMA_GGML_CMAKE_DIR").expect("llama-cpp-sys-2 did not say where ggml was installed"));
  let install = cmake_dir.parent().and_then(Path::parent).expect("ggml cmake dir should be nested");
  // Windows puts DLLs in `bin`, the others in `lib`/`lib64`.
  for dir in ["bin", "lib", "lib64"] {
    copy_shared_libraries(&install.join(dir), &staging);
  }
  let backends = PathBuf::from(std::env::var("DEP_LLAMA_BACKENDS_DIR").expect("llama-cpp-sys-2 did not say where the backends are"));
  copy_shared_libraries(&backends, &staging.join("backends"));
}

/// Splits a shared library's file name into its name and the number of version
/// parts it carries: `libllama.so.0.4.1` and `libllama.0.4.1.dylib` are
/// `("libllama", 3)`, `libllama.so.0` is `("libllama", 1)`, `libggml-vulkan.so`
/// and `ggml.dll` have none. `None` if it isn't a shared library.
fn split_library_name(name: &str) -> Option<(&str, usize)> {
  let (stem, versions) = if let Some(base) = name.strip_suffix(".dylib") {
    base.split_once('.').unwrap_or((base, ""))
  } else if let Some((stem, rest)) = name.split_once(".so") {
    (stem, rest.strip_prefix('.').unwrap_or(rest))
  } else {
    (name.strip_suffix(".dll")?, "")
  };
  Some((stem, if versions.is_empty() { 0 } else { versions.split('.').count() }))
}

/// Copies llama.cpp's libraries, once each. On Linux and macOS CMake installs a
/// library under three names: the file (`libllama.so.0.4.1`), a link named after
/// its soname (`libllama.so.0`), which is what the executable and the other
/// libraries ask the loader for, and a link for the linker (`libllama.so`).
/// Tauri bundles a link as a file of its own, so bundling all three would make
/// the installer three times as big. Only the soname is needed: it is the name
/// with the fewest version parts, apart from the bare one. Libraries that only
/// have one name (`ggml.dll`, the backend modules) are copied as they are.
fn copy_shared_libraries(from: &Path, to: &Path) {
  let Ok(entries) = std::fs::read_dir(from) else { return };
  let files: Vec<_> = entries.flatten().filter(|e| e.path().is_file()).collect();
  let names: Vec<String> = files.iter().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
  // The part count to keep for each library; no version parts sorts last.
  let rank = |parts: usize| if parts == 0 { usize::MAX } else { parts };
  let mut wanted = std::collections::HashMap::new();
  for (stem, parts) in names.iter().filter_map(|name| split_library_name(name)) {
    let best = wanted.entry(stem).or_insert(parts);
    if rank(parts) < rank(*best) {
      *best = parts;
    }
  }
  for (entry, name) in files.iter().zip(&names) {
    let Some((stem, parts)) = split_library_name(name) else { continue };
    let target = to.join(name);
    if parts != wanted[stem] {
      // An earlier build staged this name too; the bundle shouldn't take it.
      let _ = std::fs::remove_file(&target);
      continue;
    }
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

fn write_if_different(path: &Path, contents: &[u8]) {
  if std::fs::read(path).ok().as_deref() != Some(contents) {
    std::fs::write(path, contents).unwrap();
  }
}
