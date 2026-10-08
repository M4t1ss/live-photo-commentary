//! Development tasks for this workspace, run with `cargo xtask TASK`.
//!
//! `cargo xtask package-helper OUT_DIR [--target TRIPLE] [--tauri]`
//!
//! Builds `kokoro-espeak` in release mode and copies it, with eSpeak's
//! `espeak-ng-data` folder, into `OUT_DIR`. With `--tauri`, the helper is
//! named `kokoro-espeak-<target triple>`, as Tauri sidecars require.
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

const USAGE: &str = "usage: cargo xtask package-helper OUT_DIR [--target TRIPLE] [--tauri]";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("package-helper") => package_helper(&args[1..]),
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
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("xtask is in the workspace");
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
