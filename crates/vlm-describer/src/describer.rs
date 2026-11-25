//! Describes screenshots with a vision-language model, keeping a history of
//! its comments for continuity. Ported from the `Describer` class in
//! live-photo-commentary's `describer.py`.

use std::time::Instant;

use image::DynamicImage;
use regex::Regex;

use crate::backend::Backend;
use crate::error::Result;
use crate::prompts::Prompts;

/// Polled while a request is in flight; when it returns true the request is
/// abandoned and the call returns [`Error::Cancelled`](crate::Error::Cancelled).
/// A local model checks it before every token; a cloud request is dropped
/// within about 100 ms.
pub type CancelFn<'a> = &'a (dyn Fn() -> bool + Sync);

/// Per-call settings of [`Describer::describe_with`].
#[derive(Clone, Copy, Default)]
pub struct DescribeOptions<'a> {
    /// See [`CancelFn`]. `None`: never cancelled.
    pub cancel: Option<CancelFn<'a>>,
    /// What the user said, if they did: [`Prompts::voice_prompt`] with it is
    /// added to the prompt, and once the comment has been made, `User said:`
    /// and the speech go into the history just before it. `None`: a normal
    /// description.
    pub user_text: Option<&'a str>,
}

pub struct Describer {
    backend: Backend,
    pub prompts: Prompts,
    /// How many past comments to include in the prompt; 0 for none. When
    /// the history reaches this size, it is summarized first.
    pub max_history_size: usize,
    /// How many of the newest comments to keep as they are when summarizing.
    pub min_history_size: usize,
    history: Vec<String>,
    response_re: Option<Regex>,
}

impl Describer {
    /// A describer with the default prompts and no history.
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            prompts: Prompts::default(),
            max_history_size: 0,
            min_history_size: 0,
            history: Vec::new(),
            response_re: None,
        }
    }

    /// Extracts the comment from each response with a regex, for models that
    /// wrap their answer in other output. The comment is the group named
    /// `text`, or else the first group, or else the whole match. `.` matches
    /// newlines too.
    pub fn with_response_re(mut self, pattern: &str) -> Result<Self> {
        self.response_re = Some(Regex::new(&format!("(?s){pattern}"))?);
        Ok(self)
    }

    /// Comments on `current`. If there is a `previous` screenshot, the model
    /// is asked what changed, with the comment history for context.
    pub async fn describe(
        &mut self,
        current: &DynamicImage,
        previous: Option<&DynamicImage>,
    ) -> Result<String> {
        self.describe_with(current, previous, &DescribeOptions::default()).await
    }

    /// Like [`describe`](Self::describe), with per-call options. If the
    /// cancel function fires, returns [`Error::Cancelled`](crate::Error::Cancelled) and leaves the
    /// history as it was (a history compaction that already finished stays).
    pub async fn describe_with(
        &mut self,
        current: &DynamicImage,
        previous: Option<&DynamicImage>,
        options: &DescribeOptions<'_>,
    ) -> Result<String> {
        let never = || false;
        let cancel: CancelFn<'_> = options.cancel.unwrap_or(&never);
        let mut images = vec![current];
        let user_prompt = match previous {
            Some(previous) => {
                images.push(previous);
                if self.max_history_size > 0 {
                    if self.history.len() >= self.max_history_size {
                        self.compact_history(cancel).await?;
                    }
                    let mut lines = vec![self.prompts.history_prompt.as_str()];
                    lines.extend(self.history.iter().map(String::as_str));
                    lines.extend(["---", &self.prompts.prompt]);
                    lines.join("\n")
                } else {
                    self.prompts.prompt.clone()
                }
            }
            None => self.prompts.first_prompt.clone(),
        };
        let user_prompt = match options.user_text {
            Some(speech) => format!("{user_prompt}\n\n{}", self.prompts.voice_text(speech)),
            None => user_prompt,
        };

        let response = self
            .run_prompt(&user_prompt, &images, Some(&self.prompts.system_prompt), cancel)
            .await?;
        let comment = self.extract_comment(response);
        if let Some(speech) = options.user_text {
            self.history.push(format!("User said: {speech}"));
        }
        self.history.push(comment.clone());
        Ok(comment)
    }

    /// Text-only generation, e.g. for greetings. Uses the describer's system
    /// prompt unless another is given. Doesn't touch the history.
    pub async fn generate(&self, prompt: &str, system_prompt: Option<&str>) -> Result<String> {
        let system_prompt = system_prompt.unwrap_or(&self.prompts.system_prompt);
        self.run_prompt(prompt, &[], Some(system_prompt), &|| false).await
    }

    /// The comments so far, oldest first.
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Forgets the comment history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Replaces the history with a summary, keeping the newest
    /// `min_history_size` comments as they are.
    async fn compact_history(&mut self, cancel: CancelFn<'_>) -> Result<()> {
        let keep = self.min_history_size.min(self.history.len());
        let split = self.history.len() - keep;
        let mut lines = vec![self.prompts.compact_prompt.as_str()];
        lines.extend(self.history[..split].iter().map(String::as_str));
        let summary = self
            .run_prompt(&lines.join("\n"), &[], Some(&self.prompts.system_prompt), cancel)
            .await?;
        let kept = self.history.split_off(split);
        self.history = std::iter::once(summary).chain(kept).collect();
        Ok(())
    }

    async fn run_prompt(
        &self,
        user_prompt: &str,
        images: &[&DynamicImage],
        system_prompt: Option<&str>,
        cancel: CancelFn<'_>,
    ) -> Result<String> {
        log::debug!("system_prompt:\n{}", system_prompt.unwrap_or(""));
        log::debug!("user_prompt:\n{user_prompt}");
        let start = Instant::now();
        let response = self.backend.prompt_model(user_prompt, images, system_prompt, cancel).await?;
        log::info!("response ({:.1}s):\n{response}", start.elapsed().as_secs_f64());
        Ok(response)
    }

    fn extract_comment(&self, response: String) -> String {
        let Some(re) = &self.response_re else {
            return response;
        };
        let Some(captures) = re.captures(&response) else {
            return response;
        };
        let group = if re.capture_names().any(|name| name == Some("text")) {
            captures.name("text")
        } else {
            captures.get(1).or_else(|| captures.get(0))
        };
        group.map_or(String::new(), |m| m.as_str().to_string())
    }
}
