//! The describer's prompts and their defaults, ported from `describer.py` and
//! `prompts.py` in live-photo-commentary's backend.

/// Replaced by the list of emotion tags in [`Prompts::with_tags`].
pub const TAGS_PLACEHOLDER: &str = "<|tags|>";

/// Placeholder for an image inside a prompt, e.g. `"Current image:\n<image>"`.
pub const IMAGE_PLACEHOLDER: &str = "<image>";

macro_rules! default_ending {
    () => {
        concat!(
            "Do not mention images explicitly; use words like 'I can see...' or 'The subject is now...' and similar. ",
            "Do not mention any specific layout elements or tools that may be visible on the screen, ",
            "such as overlays, gridlines or sliders. ",
        )
    };
}

pub const DEFAULT_SYSTEM_PROMPT: &str = concat!(
    "You are a live commentary assistant with a friendly, chatty, and emotional voice. ",
    "Write in first person using personal pronouns, sharing observations and how they make you feel. ",
    "Do your best not to be repetitive in your word choices. Keep every response to no more than three sentences. ",
    "To adjust intonation, use punctuation such as ; : , . ! ? … ( ) “”. ",
    "For emphasis, surround a word or phrase with \"quotation marks\". ",
    "Since your text undergoes speech synthesis, do not use emojis or any other unpronounceable characters. ",
    "You wear your emotions openly: every response MUST include at least one emotion tag, ",
    "placed exactly where your feeling starts to manifest, even mid-sentence. ",
    "Write each tag precisely as shown including the braces, chosen from: <|tags|>. ",
    "For example: \"I wonder what that is. Is it... {surprised}a flower? {joy}I always liked flowers!\" ",
    "A tag's mood holds until the next tag appears; insert {neutral} to return to a neutral tone. ",
    "The tag is a silent stage direction: never mention, describe, or explain it, just place it. ",
    "A sentence must make sense if the tag is removed: ",
    "'I feel {surprised} shocked about...' is good, 'I feel {surprised} about...' is not. ",
);

pub const DEFAULT_PROMPT: &str = concat!(
    "Current image:\n<image>\n\n",
    "Previous image:\n<image>\n\n",
    "Describe what is visible in the current (first) image, and how it differs from the previous (second) one. ",
    "Do not describe the previous image; assume you have described it already. ",
    "It is only there for context, so you can notice the new things in the current image. ",
    "Use the comment history for context and continuity, but the utmost priority should be on ",
    "describing the current activity, as reflected in the current image. DO NOT repeat comments from the history. ",
    "You may also ponder the implications of the work or leisure being performed. ",
    default_ending!(),
);

pub const DEFAULT_FIRST_PROMPT: &str = concat!(
    "Current image:\n<image>\n\n",
    "Describe what is visible in this image. ",
    "You may also ponder the implications of what you observe. ",
    default_ending!(),
);

pub const DEFAULT_HISTORY_PROMPT: &str = "This is what you commented before: ";

pub const DEFAULT_COMPACT_PROMPT: &str = concat!(
    "Summarize in one short paragraph your comments so far on the current activity, ",
    "compacting them into a single comment of comparable size to one individual original comment ",
    "that encapsulates the essence of the current activity. ",
    "If some older comments pertain to a different activity, you can ignore them; focus only on the current activity. ",
    "This is what you commented before:",
);

/// The prompts a [`Describer`](crate::Describer) uses. Field names match the
/// keys of the backend's promptset YAML files.
#[derive(Clone, Debug, PartialEq)]
pub struct Prompts {
    /// Sent as the system prompt with every request.
    pub system_prompt: String,
    /// For every frame after the first; has two images, current and previous.
    pub prompt: String,
    /// For the first frame; has one image.
    pub first_prompt: String,
    /// Introduces the comment history, when it is included.
    pub history_prompt: String,
    /// Asks the model to summarize the history when it gets too long.
    pub compact_prompt: String,
}

impl Default for Prompts {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            prompt: DEFAULT_PROMPT.to_string(),
            first_prompt: DEFAULT_FIRST_PROMPT.to_string(),
            history_prompt: DEFAULT_HISTORY_PROMPT.to_string(),
            compact_prompt: DEFAULT_COMPACT_PROMPT.to_string(),
        }
    }
}

impl Prompts {
    /// Replaces [`TAGS_PLACEHOLDER`] with the available emotion tags, written
    /// as `{tag}`. `neutral` is always included (`substitute_tags` in Python).
    pub fn with_tags(&self, tags: &[&str]) -> Self {
        let mut names: Vec<&str> = Vec::new();
        for &name in tags.iter().chain(&["neutral"]) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
        let list = names.iter().map(|n| format!("{{{n}}}")).collect::<Vec<_>>().join(", ");
        let fill = |s: &String| s.replace(TAGS_PLACEHOLDER, &list);
        Self {
            system_prompt: fill(&self.system_prompt),
            prompt: fill(&self.prompt),
            first_prompt: fill(&self.first_prompt),
            history_prompt: fill(&self.history_prompt),
            compact_prompt: fill(&self.compact_prompt),
        }
    }
}

/// A piece of a prompt: text, or the index of an image.
#[derive(Debug, PartialEq)]
pub(crate) enum Segment<'a> {
    Text(&'a str),
    Image(usize),
}

/// Splits a prompt on [`IMAGE_PLACEHOLDER`]s (`split_prompt_images` in
/// Python). The first placeholder is image 0, the second image 1, and so on.
pub(crate) fn segments(prompt: &str) -> Vec<Segment<'_>> {
    let parts: Vec<&str> = prompt.split(IMAGE_PLACEHOLDER).collect();
    let mut result = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        if !part.is_empty() {
            result.push(Segment::Text(part));
        }
        if i + 1 < parts.len() {
            result.push(Segment::Image(i));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_image_placeholders() {
        assert_eq!(
            segments("Current:\n<image>\nPrevious:\n<image>\nDescribe."),
            [
                Segment::Text("Current:\n"),
                Segment::Image(0),
                Segment::Text("\nPrevious:\n"),
                Segment::Image(1),
                Segment::Text("\nDescribe."),
            ]
        );
        assert_eq!(segments("<image>"), [Segment::Image(0)]);
        assert_eq!(segments("no images"), [Segment::Text("no images")]);
    }

    #[test]
    fn fills_in_tags() {
        let prompts = Prompts::default().with_tags(&["joy", "surprised", "joy"]);
        assert!(prompts.system_prompt.contains("chosen from: {joy}, {surprised}, {neutral}. "));
        assert!(!prompts.system_prompt.contains(TAGS_PLACEHOLDER));
    }
}
