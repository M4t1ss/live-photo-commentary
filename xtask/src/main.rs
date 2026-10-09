//! Development tasks for this workspace, run with `cargo xtask TASK`.
//!
//! `cargo xtask package-helper OUT_DIR [--target TRIPLE] [--tauri]`
//!
//! Builds `kokoro-espeak` in release mode and copies it, with eSpeak's
//! `espeak-ng-data` folder, into `OUT_DIR`. With `--tauri`, the helper is
//! named `kokoro-espeak-<target triple>`, as Tauri sidecars require.
//!
//! `cargo xtask package-cuda-backend OUT_DIR [--release]`
//!
//! Builds the app with `dynamic-backends,cuda` and zips the CUDA backend
//! library with NVIDIA's cuBLAS libraries and the CUDA Toolkit's `EULA.txt`
//! (which allows redistributing them; from `CUDA_PATH`) into
//! `OUT_DIR/cuda-backend-<os>-x64.zip`, plus a `.sha256` file with the
//! checksum the app is built with (`CUDA_BACKEND_SHA256`). The app downloads
//! this archive from Settings. Set `CMAKE_CUDA_ARCHITECTURES` to build for
//! fewer GPU generations than the default (all), e.g. `86`.
//!
//! `espeak-ng-data` is created inside espeak-rs-sys's build directory, whose
//! name has a hash in it. Cargo reports that directory in its JSON messages
//! (`build-script-executed`), so it's taken from there.

use std::error::Error;
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;
use sha2::{Digest, Sha256};

const USAGE: &str = "usage: cargo xtask package-helper OUT_DIR [--target TRIPLE] [--tauri]
       cargo xtask package-cuda-backend OUT_DIR [--release]";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("package-helper") => package_helper(&args[1..]),
        Some("package-cuda-backend") => package_cuda_backend(&args[1..]),
        _ => Err(USAGE.into()),
    }
}

fn package_helper(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut out_dir = None;
    let mut target = None;
    let mut tauri = false;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--target" => target = Some(args.next().ok_or(USAGE)?.clone()),
            "--tauri" => tauri = true,
            _ if out_dir.is_none() => out_dir = Some(PathBuf::from(arg)),
            _ => return Err(format!("unexpected argument {arg:?}\n{USAGE}").into()),
        }
    }
    // Absolute, since the build below runs in the workspace's folder.
    let out_dir = std::path::absolute(out_dir.ok_or(USAGE)?)?;

    let (helper, data) = build_helper(target.as_deref())?;

    fs::create_dir_all(&out_dir)?;
    let helper_name = if tauri {
        let triple = match target {
            Some(triple) => triple,
            None => host_triple()?,
        };
        let mut name = format!("kokoro-espeak-{triple}");
        if let Some(extension) = helper.extension() {
            name = format!("{name}.{}", extension.to_string_lossy());
        }
        name.into()
    } else {
        helper.file_name().ok_or("helper has no file name")?.to_owned()
    };
    fs::copy(&helper, out_dir.join(&helper_name))?;
    let data_destination = out_dir.join("espeak-ng-data");
    if data_destination.exists() {
        fs::remove_dir_all(&data_destination)?;
    }
    copy_dir(&data, &data_destination)?;
    println!(
        "copied {} and espeak-ng-data to {}",
        helper_name.to_string_lossy(),
        out_dir.display()
    );
    Ok(())
}

/// Builds the helper, returning its path and that of `espeak-ng-data`.
fn build_helper(target: Option<&str>) -> Result<(PathBuf, PathBuf), Box<dyn Error>> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut command = Command::new(cargo);
    // Run in this workspace even when called from another project, so that
    // `-p` finds the helper and Cargo reads this workspace's
    // `.cargo/config.toml` (which sets `/utf-8` for eSpeak NG).
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("xtask should be in the workspace");
    command.current_dir(workspace);
    command.args(["build", "--release", "-p", "kokoro-espeak"]);
    // Compiler messages still go to stderr; stdout gets Cargo's JSON.
    command.arg("--message-format=json-render-diagnostics");
    if let Some(target) = target {
        command.args(["--target", target]);
    }
    let output = command.stderr(Stdio::inherit()).output()?;
    if !output.status.success() {
        return Err("building kokoro-espeak failed".into());
    }

    let mut helper = None;
    let mut data = None;
    for line in output.stdout.lines() {
        let message: Value = serde_json::from_str(&line?)?;
        match message["reason"].as_str() {
            Some("compiler-artifact") if message["target"]["name"] == "kokoro-espeak" => {
                helper = message["executable"].as_str().map(PathBuf::from);
            }
            Some("build-script-executed")
                if message["package_id"].as_str().is_some_and(|id| id.contains("espeak-rs-sys")) =>
            {
                let out_dir = message["out_dir"].as_str().ok_or("no out_dir")?;
                data = Some(Path::new(out_dir).join("share").join("espeak-ng-data"));
            }
            _ => {}
        }
    }
    let helper = helper.ok_or("Cargo didn't report the kokoro-espeak executable")?;
    let data = data.ok_or("Cargo didn't report espeak-rs-sys's build directory")?;
    if !data.is_dir() {
        return Err(format!("{} doesn't exist", data.display()).into());
    }
    Ok((helper, data))
}

/// The platform Rust builds for by default, e.g. `x86_64-pc-windows-msvc`.
fn host_triple() -> Result<String, Box<dyn Error>> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let output = Command::new(rustc).arg("-vV").output()?;
    String::from_utf8(output.stdout)?
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_string)
        .ok_or_else(|| "rustc -vV printed no host".into())
}

fn copy_dir(source: &Path, destination: &Path) -> std::io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn package_cuda_backend(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut out_dir = None;
    let mut release = false;
    for arg in args {
        match arg.as_str() {
            "--release" => release = true,
            _ if out_dir.is_none() => out_dir = Some(PathBuf::from(arg)),
            _ => return Err(format!("unexpected argument {arg:?}
{USAGE}").into()),
        }
    }
    let out_dir = std::path::absolute(out_dir.ok_or(USAGE)?)?;
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("xtask should be in the workspace");

    // The app's build script copies the backend library to `llama-cuda/backends`.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut build = Command::new(cargo);
    build.current_dir(workspace).args(["build", "-p", "app", "--features", "dynamic-backends,cuda"]);
    if release {
        build.arg("--release");
    }
    if !build.status()?.success() {
        return Err("building the app with CUDA failed".into());
    }
    let (os, backend_name) = if cfg!(windows) {
        ("windows", "ggml-cuda.dll")
    } else {
        ("linux", "libggml-cuda.so")
    };
    let backend = workspace.join("src-tauri/resources/llama-cuda/backends").join(backend_name);
    if !backend.is_file() {
        return Err(format!("{} was not built", backend.display()).into());
    }

    fs::create_dir_all(&out_dir)?;
    // Linux: the library finds cuBLAS next to itself only with an `$ORIGIN`
    // runpath, which llama.cpp's build doesn't set. Patch a copy.
    let stage = out_dir.join("stage");
    let backend = if cfg!(windows) { backend } else { with_origin_runpath(&backend, &stage)? };
    let mut files = vec![backend];
    files.extend(cublas_libraries()?);
    // The licence under which NVIDIA lets us redistribute cuBLAS; it travels
    // with the libraries (and reproduces the notices of code inside cuBLAS).
    let eula = cuda_toolkit()?.join("EULA.txt");
    if !eula.is_file() {
        return Err(format!("{} not found; NVIDIA's licence must ship with cuBLAS", eula.display()).into());
    }
    files.push(eula);

    let archive = out_dir.join(format!("cuda-backend-{os}-x64.zip"));
    let mut zip = zip::ZipWriter::new(fs::File::create(&archive)?);
    for file in &files {
        let name = file.file_name().ok_or("file without a name")?.to_string_lossy().into_owned();
        let name = if name == "EULA.txt" { "NVIDIA-CUDA-EULA.txt".to_string() } else { name };
        zip.start_file(name, zip::write::SimpleFileOptions::default().large_file(true))?;
        std::io::copy(&mut fs::File::open(file)?, &mut zip)?;
    }
    zip.finish()?;
    let _ = fs::remove_dir_all(&stage);

    let digest = Sha256::digest(fs::read(&archive)?);
    let checksum: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    fs::write(archive.with_extension("zip.sha256"), format!("{checksum}
"))?;
    println!("{} ({} files)
sha256 {checksum}", archive.display(), files.len());
    Ok(())
}

/// The CUDA Toolkit's folder.
fn cuda_toolkit() -> Result<PathBuf, Box<dyn Error>> {
    std::env::var_os("CUDA_PATH")
        .map(PathBuf::from)
        .or_else(|| cfg!(unix).then(|| PathBuf::from("/usr/local/cuda")))
        .ok_or_else(|| "CUDA_PATH is not set".into())
}

/// cuBLAS and cuBLASLt from the CUDA Toolkit, which the backend needs at run
/// time and which NVIDIA lets applications redistribute.
fn cublas_libraries() -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let cuda = cuda_toolkit()?;
    let folders = if cfg!(windows) {
        vec![cuda.join("bin").join("x64"), cuda.join("bin")]
    } else {
        vec![cuda.join("lib64"), cuda.join("targets/x86_64-linux/lib")]
    };
    // `cublas64_13.dll`; `libcublas.so.13` (not the `.so` link or `.so.13.1.2`).
    let is_wanted = |name: &str| {
        let name = name.to_ascii_lowercase();
        if cfg!(windows) {
            (name.starts_with("cublas64_") || name.starts_with("cublaslt64_")) && name.ends_with(".dll")
        } else {
            ["libcublas.so.", "libcublaslt.so."]
                .iter()
                .any(|prefix| name.strip_prefix(prefix).is_some_and(|rest| rest.chars().all(|c| c.is_ascii_digit())))
        }
    };
    for folder in folders {
        let found: Vec<PathBuf> = fs::read_dir(&folder)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| is_wanted(&entry.file_name().to_string_lossy()))
            .map(|entry| entry.path())
            .collect();
        if found.len() == 2 {
            return Ok(found);
        }
    }
    Err(format!("cuBLAS and cuBLASLt not found under {}", cuda.display()).into())
}

/// Copies `library` to `stage` and sets its runpath to `$ORIGIN`, so that the
/// dynamic loader looks for the libraries it needs in its own folder. Needs
/// `patchelf`.
#[cfg_attr(windows, allow(dead_code))]
fn with_origin_runpath(library: &Path, stage: &Path) -> Result<PathBuf, Box<dyn Error>> {
    fs::create_dir_all(stage)?;
    let copy = stage.join(library.file_name().ok_or("library without a name")?);
    fs::copy(library, &copy)?;
    let status = Command::new("patchelf").arg("--set-rpath").arg("$ORIGIN").arg(&copy).status()?;
    if !status.success() {
        return Err("patchelf failed".into());
    }
    Ok(copy)
}
