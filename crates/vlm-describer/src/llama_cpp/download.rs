//! Downloading local models from Hugging Face, with progress reports.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use hf_hub::HFClient;
use hf_hub::HFRepository;
use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};
use hf_hub::repository::repo_type::RepoTypeModel;

use crate::error::{Error, Result};

/// Called during a download with the file name, the bytes downloaded so far
/// and the file's size, at most once per percent.
pub type DownloadProgress = Arc<dyn Fn(&str, u64, u64) + Send + Sync>;

/// Downloads a GGUF model file and its vision projector (`mmproj`) file from
/// the Hugging Face repo `repo` (`owner/name`), returning their local paths
/// `(model, mmproj)`. Files already in the cache aren't downloaded again,
/// and no network is needed then.
///
/// The cache is Hugging Face's usual one, shared with Python: `HF_HOME` and
/// similar variables are respected, and without them it is
/// `~/.cache/huggingface/hub`.
pub async fn download_model(
    repo: &str,
    model_file: &str,
    mmproj_file: &str,
    progress: Option<DownloadProgress>,
) -> Result<(PathBuf, PathBuf)> {
    let (owner, name) = repo.split_once('/').ok_or_else(|| Error::InvalidRepo(repo.to_string()))?;
    let mut builder = HFClient::builder();
    if let Some(cache_dir) = hf_cache_dir() {
        builder = builder.cache_dir(cache_dir);
    }
    let client = builder
        .build()
        .map_err(|source| Error::Download { file: repo.to_string(), source })?;
    let repo = client.model(owner, name);
    let model = download_file(&repo, model_file, progress.clone()).await?;
    let mmproj = download_file(&repo, mmproj_file, progress).await?;
    Ok((model, mmproj))
}

/// Uses the cached file if there is one, without checking for updates.
/// On Windows, hf-hub 1.0 copies a cached file again on every download call
/// (it can't make symlinks), so avoid the call when possible.
async fn download_file(
    repo: &HFRepository<RepoTypeModel>,
    file: &str,
    progress: Option<DownloadProgress>,
) -> Result<PathBuf> {
    let cached = repo.download_file().filename(file).local_files_only(true).send().await;
    if let Ok(path) = cached {
        return Ok(path);
    }
    let reporter = progress.map(|callback| {
        Progress::new(Reporter {
            file: file.to_string(),
            callback,
            last_percent: AtomicU64::new(u64::MAX),
        })
    });
    repo.download_file()
        .filename(file)
        .maybe_progress(reporter)
        .send()
        .await
        .map_err(|source| Error::Download { file: file.to_string(), source })
}

/// Turns hf-hub's events for one file into [`DownloadProgress`] calls.
struct Reporter {
    file: String,
    callback: DownloadProgress,
    /// The percentage last reported (`u64::MAX` before the first report).
    last_percent: AtomicU64,
}

impl ProgressHandler for Reporter {
    fn on_progress(&self, event: &ProgressEvent) {
        // Plain HTTP downloads report per file; Xet downloads report totals
        // for the whole batch, which is just this one file.
        let (done, total) = match event {
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                match files.iter().find(|f| f.filename == self.file) {
                    Some(file) => (file.bytes_completed, file.total_bytes),
                    None => return,
                }
            }
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            }) => (*bytes_completed, *total_bytes),
            _ => return,
        };
        if total == 0 {
            return;
        }
        let percent = done.min(total) * 100 / total;
        if self.last_percent.swap(percent, Ordering::Relaxed) != percent {
            (self.callback)(&self.file, done, total);
        }
    }
}

/// The Hugging Face cache folder, if hf-hub would get it wrong.
///
/// Without `HF_HOME` and similar variables, hf-hub 1.0 uses `$HOME/.cache`,
/// and `/tmp/.cache` if `HOME` is unset, as it normally is on Windows. Python's
/// `huggingface_hub` uses the user's home folder instead, so use that, which
/// also shares the cache with Python.
fn hf_cache_dir() -> Option<PathBuf> {
    let variables = ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "HOME"];
    if variables.iter().any(|v| std::env::var_os(v).is_some()) {
        return None;
    }
    std::env::home_dir().map(|home| home.join(".cache").join("huggingface").join("hub"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::progress::{FileProgress, FileStatus};
    use std::sync::Mutex;

    #[test]
    fn reports_each_percent_of_its_own_file_once() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&calls);
        let reporter = Reporter {
            file: "model.gguf".into(),
            callback: Arc::new(move |file, done, total| {
                recorded.lock().unwrap().push((file.to_string(), done, total));
            }),
            last_percent: AtomicU64::new(u64::MAX),
        };
        let file = |filename: &str, done| {
            ProgressEvent::Download(DownloadEvent::Progress {
                files: vec![FileProgress {
                    filename: filename.into(),
                    bytes_completed: done,
                    total_bytes: 1000,
                    status: FileStatus::InProgress,
                }],
            })
        };
        let aggregate = |done| {
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed: done,
                total_bytes: 1000,
                bytes_per_sec: None,
            })
        };
        for event in [file("model.gguf", 0), file("model.gguf", 5), file("other", 500), aggregate(10), aggregate(1000)] {
            reporter.on_progress(&event);
        }
        let calls = calls.lock().unwrap();
        let expected = [0, 10, 1000].map(|done| ("model.gguf".to_string(), done, 1000));
        assert_eq!(*calls, expected);
    }
}
