//! Finds a pre-built flash-attn v2 wheel compatible with the current CUDA/Python/platform.
//!
//! Fetches the release list from mjun0812's wheel repo on GitHub, falling back to
//! the author's hosted JSON mirror when the API is unavailable or rate-limited.

use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::sync::LazyLock;
use regex::Regex;

const REPO_OWNER: &str = "mjun0812";
const REPO_NAME: &str = "flash-attention-prebuild-wheels";
const API_BASE: &str = "https://api.github.com";
const FALLBACK_URL: &str =
    "https://mjunya.com/flash-attention-prebuild-wheels/data/releases.json";
const UA: &str = concat!("live-photo-commentary/", env!("CARGO_PKG_VERSION"));

type BoxError = Box<dyn std::error::Error + Send + Sync>;

// Groups: 1=flash_version  2=cuda_digits  3=torch_version  4=python  5=platform_tag
static WHEEL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"flash_attn(?:_3)?-(\d+\.\d+\.\d+(?:\.[a-z0-9]+)?)\+cu(\d+)torch(\d+\.\d+)(?:git[0-9a-f]+)?-cp(\d+)-(?:cp\d+t?|abi3)-(.+?)\.whl",
    )
    .unwrap()
});

#[derive(Deserialize)]
struct Release {
    assets: Vec<Asset>,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
}

pub struct FlashAttnWheel {
    pub url: String,
    /// Torch minor version this wheel was built against, e.g. `"2.7"`.
    pub torch_version: String,
    pub flash_version: String,
}

struct Candidate {
    flash_version: String,
    torch_version: String,
    url: String,
}

/// `"cu128"` → `"12.8"`, `"cu118"` → `"11.8"`.
/// CUDA major is always 2 digits in all current versions (11.x, 12.x).
fn cu_to_dotted(cu_index: &str) -> Option<String> {
    let digits = cu_index.strip_prefix("cu")?;
    if digits.len() < 3 {
        return None;
    }
    let (maj, min) = digits.split_at(2);
    Some(format!("{maj}.{min}"))
}

/// Parse a version string into a sortable tuple.
/// Handles `"2.7"`, `"2.7.4"`, `"2.7.4.post1"`.
fn version_tuple(v: &str) -> (u32, u32, u32, u32) {
    let mut parts = v.split('.');
    let maj:  u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let min:  u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let pat:  u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let post: u32 = parts.next()
        .and_then(|s| s.strip_prefix("post"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (maj, min, pat, post)
}

fn current_platform() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => "Windows x86_64",
        ("linux",   "x86_64") => "Linux x86_64",
        ("linux",  "aarch64") => "Linux arm64",
        _ => "",
    }
}

fn normalize_platform(raw: &str) -> String {
    // Dotted manylinux tags: "manylinux_2_17_x86_64.manylinux2014_x86_64" → take first segment
    let raw = if raw.starts_with("manylinux") && raw.contains('.') {
        raw.split('.').next().unwrap_or(raw)
    } else {
        raw
    };
    // "manylinux_2_17_x86_64" → "Linux x86_64"
    if raw.starts_with("manylinux") {
        let parts: Vec<&str> = raw.split('_').collect();
        if parts.len() >= 4 {
            let arch = parts[3..].join("_");
            let arch = if arch == "aarch64" { "arm64".to_string() } else { arch };
            return format!("Linux {arch}");
        }
    }
    // "win_amd64" → "Windows x86_64"
    let s = match raw.chars().next() {
        Some(c) => format!("{}{}", c.to_ascii_uppercase(), &raw[c.len_utf8()..]),
        None => return String::new(),
    };
    s.replace('_', " ")
        .replace("Win", "Windows")
        .replace("amd64", "x86_64")
        .replace("aarch64", "arm64")
}

async fn fetch_github(client: &Client) -> Result<Vec<Release>, BoxError> {
    let mut out = Vec::new();
    let mut page = 1u32;
    loop {
        let url = format!(
            "{API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/releases?page={page}&per_page=100"
        );
        let resp = client.get(&url).header("User-Agent", UA).send().await?;
        if !resp.status().is_success() {
            return Err(format!("GitHub API returned {}", resp.status()).into());
        }
        let batch: Vec<Release> = resp.json().await?;
        let done = batch.len() < 100;
        out.extend(batch);
        if done {
            break;
        }
        page += 1;
    }
    Ok(out)
}

async fn fetch_hosted(client: &Client) -> Result<Vec<Release>, BoxError> {
    let resp = client.get(FALLBACK_URL).header("User-Agent", UA).send().await?;
    if !resp.status().is_success() {
        return Err(format!("Hosted mirror returned {}", resp.status()).into());
    }
    let data: Value = resp.json().await?;
    let releases = serde_json::from_value(
        data.get("releases")
            .ok_or("missing releases key")?
            .clone(),
    )?;
    Ok(releases)
}

/// Returns the best flash-attn v2 wheel for the given CUDA index and Python version tag.
///
/// `cu_index`: e.g. `"cu128"`
/// `python_version`: e.g. `"312"` for CPython 3.12
///
/// Among all matching wheels (same python/cuda/platform), picks the highest supported torch
/// minor version — that version will be used to pin torch in the CUDA venv.  Within that torch
/// level, picks the highest flash-attn release.  Returns `None` if the release list is
/// unreachable or no matching wheel exists (wrong CUDA, unsupported platform, etc.).
pub async fn find_flash_attn_wheel(
    cu_index: &str,
    python_version: &str,
) -> Option<FlashAttnWheel> {
    let cuda_dotted = cu_to_dotted(cu_index)?;
    let platform = current_platform();
    if platform.is_empty() {
        return None;
    }

    let client = Client::new();
    let releases = match fetch_github(&client).await {
        Ok(r) => r,
        Err(e) => {
            log::warn!("flash_attn: GitHub fetch failed ({e}); trying hosted mirror");
            fetch_hosted(&client).await
                .map_err(|e| log::warn!("flash_attn: hosted mirror also failed ({e})"))
                .ok()?
        }
    };

    let mut candidates: Vec<Candidate> = Vec::new();

    for release in &releases {
        for asset in &release.assets {
            if !asset.name.ends_with(".whl") || asset.name.starts_with("flash_attn_3-") {
                continue;
            }
            let caps = match WHEEL_RE.captures(&asset.name) {
                Some(c) => c,
                None => continue,
            };
            let (Some(flash_m), Some(cuda_m), Some(torch_m), Some(py_m), Some(plat_m)) = (
                caps.get(1), caps.get(2), caps.get(3), caps.get(4), caps.get(5),
            ) else {
                continue;
            };

            let cuda_digits = cuda_m.as_str();
            if cuda_digits.len() < 3 {
                continue;
            }
            let (maj, min) = cuda_digits.split_at(2);
            if format!("{maj}.{min}") != cuda_dotted {
                continue;
            }
            if py_m.as_str() != python_version {
                continue;
            }
            if normalize_platform(plat_m.as_str()) != platform {
                continue;
            }

            candidates.push(Candidate {
                flash_version: flash_m.as_str().to_string(),
                torch_version: torch_m.as_str().to_string(),
                url: asset.browser_download_url.clone(),
            });
        }
    }

    // Highest supported torch minor version — this becomes the pinned torch in pyproject.toml.
    // Use tuple comparison so "2.10" > "2.9" (string ordering would get it wrong).
    let best_torch = candidates
        .iter()
        .map(|c| c.torch_version.as_str())
        .max_by_key(|v| version_tuple(v))?
        .to_string();

    // Highest flash-attn release at that torch level.
    candidates
        .into_iter()
        .filter(|c| c.torch_version == best_torch)
        .max_by_key(|c| version_tuple(&c.flash_version))
        .map(|c| FlashAttnWheel {
            url: c.url,
            torch_version: c.torch_version,
            flash_version: c.flash_version,
        })
}
