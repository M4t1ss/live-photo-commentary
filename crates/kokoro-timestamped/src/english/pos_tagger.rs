//! Part-of-speech tagger: a port of NLTK's averaged perceptron tagger
//! (`nltk.tag.perceptron.PerceptronTagger.tag`), producing Penn Treebank tags.
//!
//! Ported from NLTK (Apache-2.0, NLTK Project), itself a port of
//! textblob-aptagger (MIT, Copyright 2013 Matthew Honnibal); see `NOTICE.md`.
//! Only tagging is ported, not training.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::embedded::decompress_xz;

static TAGGER: LazyLock<PerceptronTagger> = LazyLock::new(|| {
    // 1 MB instead of 5.7 MB; decompressing takes under 0.1 s in release builds.
    let weights = decompress_xz(include_bytes!("../../data/tagger/weights.json.xz"));
    PerceptronTagger {
        weights: serde_json::from_slice(&weights).expect("tagger weights are valid JSON"),
        tagdict: serde_json::from_str(include_str!("../../data/tagger/tags.json"))
            .expect("tagger tag dictionary is valid JSON"),
        classes: include_str!("../../data/tagger/classes.txt")
            .lines()
            .map(|line| line.trim().to_string())
            .collect(),
    }
});

struct PerceptronTagger {
    /// feature -> tag -> weight
    weights: HashMap<String, HashMap<String, f32>>,
    /// Words that always get the same tag.
    tagdict: HashMap<String, String>,
    classes: Vec<String>,
}

/// Tags tokenized text. Tokens should be split like the Penn Treebank:
/// punctuation separate from words, and `don't` as `do` + `n't`.
pub fn tag(tokens: &[&str]) -> Vec<String> {
    TAGGER.tag(tokens)
}

impl PerceptronTagger {
    fn tag(&self, tokens: &[&str]) -> Vec<String> {
        let mut context = vec!["-START-".to_string(), "-START2-".to_string()];
        context.extend(tokens.iter().map(|t| normalize(t)));
        context.extend(["-END-".to_string(), "-END2-".to_string()]);

        let mut prev = "-START-".to_string();
        let mut prev2 = "-START2-".to_string();
        let mut output = Vec::with_capacity(tokens.len());
        for (i, &word) in tokens.iter().enumerate() {
            let tag = match self.tagdict.get(word) {
                Some(tag) => tag.clone(),
                None => self.predict(&features(i + 2, word, &context, &prev, &prev2)),
            };
            prev2 = std::mem::replace(&mut prev, tag.clone());
            output.push(tag);
        }
        output
    }

    fn predict(&self, features: &HashMap<String, u32>) -> String {
        let mut scores: HashMap<&str, f32> = HashMap::new();
        for (feature, &count) in features {
            if let Some(weights) = self.weights.get(feature) {
                for (label, weight) in weights {
                    *scores.entry(label).or_insert(0.0) += count as f32 * weight;
                }
            }
        }
        // Like NLTK, ties go to the alphabetically last label.
        let score = |label: &str| scores.get(label).copied().unwrap_or(0.0);
        self.classes
            .iter()
            .max_by(|a, b| score(a).total_cmp(&score(b)).then_with(|| a.cmp(b)))
            .expect("tagger has classes")
            .clone()
    }
}

fn normalize(word: &str) -> String {
    let first = word.chars().next();
    if word.contains('-') && first != Some('-') {
        "!HYPHEN".to_string()
    } else if word.chars().count() == 4 && word.chars().all(|c| c.is_ascii_digit()) {
        "!YEAR".to_string()
    } else if first.is_some_and(|c| c.is_ascii_digit()) {
        "!DIGITS".to_string()
    } else {
        word.to_lowercase()
    }
}

/// Python's `s[-3:]`.
fn last3(s: &str) -> String {
    let n = s.chars().count();
    s.chars().skip(n.saturating_sub(3)).collect()
}

/// `i` indexes `context`, which has two start markers before the first token.
fn features(i: usize, word: &str, context: &[String], prev: &str, prev2: &str) -> HashMap<String, u32> {
    let mut features = HashMap::new();
    let mut add = |name: String| *features.entry(name).or_insert(0) += 1;
    add("bias".to_string());
    add(format!("i suffix {}", last3(word)));
    add(format!("i pref1 {}", word.chars().next().map(String::from).unwrap_or_default()));
    add(format!("i-1 tag {prev}"));
    add(format!("i-2 tag {prev2}"));
    add(format!("i tag+i-2 tag {prev} {prev2}"));
    add(format!("i word {}", context[i]));
    add(format!("i-1 tag+i word {prev} {}", context[i]));
    add(format!("i-1 word {}", context[i - 1]));
    add(format!("i-1 suffix {}", last3(&context[i - 1])));
    add(format!("i-2 word {}", context[i - 2]));
    add(format!("i+1 word {}", context[i + 1]));
    add(format!("i+1 suffix {}", last3(&context[i + 1])));
    add(format!("i+2 word {}", context[i + 2]));
    features
}
