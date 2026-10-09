//! English additions on top of eSpeak, which does the actual phonemization.
//!
//! eSpeak often guesses heteronyms ("record" the noun vs. the verb) wrong.
//! This module picks their pronunciation from their part of speech, the way
//! misaki does, and replaces eSpeak's pronunciation in the chunk's phonemes.
//!
//! `lookup` and `parent_tag` are ported from `Lexicon.lookup` and
//! `Lexicon.get_parent_tag` in `misaki/en.py` (Apache-2.0, hexgrad), limited
//! to content-word heteronyms; see `NOTICE.md`.

mod pos_tagger;

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::text_segmentation::{Word, levenshtein, norm};

/// Punctuation that eSpeak's phonemes may contain: the helper passes
/// punctuation through.
const PUNCTUATION: &str = ";:,.!?¡¿—…\"“”(){}[]";

/// word -> tag -> phonemes. Tags are Penn Treebank tags (`VBD`, ...) or the
/// parent tags `NOUN`, `VERB`, `ADJ`, `ADV`, plus `DEFAULT`.
type HeteronymTable = HashMap<String, HashMap<String, String>>;

static US_HETERONYMS: LazyLock<HeteronymTable> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../data/us_heteronyms.json"))
        .expect("American heteronym table should be valid JSON")
});

static GB_HETERONYMS: LazyLock<HeteronymTable> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../data/gb_heteronyms.json"))
        .expect("British heteronym table should be valid JSON")
});

#[derive(Clone, Copy)]
pub enum Dialect {
    American,
    British,
}

impl Dialect {
    /// eSpeak's name for the voice.
    pub fn espeak_voice(self) -> &'static str {
        match self {
            Self::American => "en-us",
            Self::British => "en",
        }
    }

    fn heteronyms(self) -> &'static HeteronymTable {
        match self {
            Self::American => &US_HETERONYMS,
            Self::British => &GB_HETERONYMS,
        }
    }
}

pub struct Heteronym {
    /// The pronunciation for this word's part of speech.
    phonemes: &'static str,
    /// All of the word's pronunciations.
    entry: &'static HashMap<String, String>,
}

/// For each word, the pronunciation to use instead of eSpeak's, if it is a
/// heteronym.
pub fn find_heteronyms(words: &[Word], dialect: Dialect) -> Vec<Option<Heteronym>> {
    let mut tokens: Vec<String> = Vec::new();
    let mut cores = Vec::with_capacity(words.len());
    for word in words {
        let (word_tokens, core) = tokenize(&word.surface);
        cores.push(core.map(|c| (tokens.len() + c, word_tokens[c].clone())));
        tokens.extend(word_tokens);
    }
    let token_refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
    let tags = pos_tagger::tag(&token_refs);

    cores
        .into_iter()
        .map(|core| {
            let (index, text) = core?;
            let heteronym = lookup(dialect.heteronyms(), &text, &tags[index])?;
            log::debug!("heteronym {text:?}/{} -> {}", tags[index], heteronym.phonemes);
            Some(heteronym)
        })
        .collect()
}

/// Splits a whitespace-separated word into Penn Treebank-style tokens:
/// punctuation on either side becomes separate tokens, quotes become
/// `` `` `` or `''`, and clitics are split off (`don't` -> `do` + `n't`,
/// `it's` -> `it` + `'s`). Also returns the index of the word's main token.
fn tokenize(word: &str) -> (Vec<String>, Option<usize>) {
    const QUOTES: [char; 5] = ['"', '\'', '“', '”', '‘'];
    let start = word.find(char::is_alphanumeric).unwrap_or(word.len());
    let end = word
        .rfind(char::is_alphanumeric)
        .map_or(start, |i| i + word[i..].chars().next().unwrap().len_utf8());

    let mut tokens: Vec<String> = word[..start]
        .chars()
        .map(|c| if QUOTES.contains(&c) { "``".to_string() } else { c.to_string() })
        .collect();
    let mut core_index = None;
    if start < end {
        let core = &word[start..end];
        core_index = Some(tokens.len());
        let lower = core.to_lowercase();
        let clitic_at = if lower.ends_with("n't") && core.len() > 3 {
            Some(core.len() - 3)
        } else {
            ["'s", "'re", "'ve", "'ll", "'d", "'m"]
                .iter()
                .find(|c| lower.ends_with(*c) && core.len() > c.len())
                .map(|c| core.len() - c.len())
        };
        match clitic_at {
            Some(at) => tokens.extend([core[..at].to_string(), core[at..].to_string()]),
            None => tokens.push(core.to_string()),
        }
    }
    tokens.extend(word[end..].chars().map(|c| {
        if QUOTES.contains(&c) || c == '’' { "''".to_string() } else { c.to_string() }
    }));
    (tokens, core_index)
}

/// misaki's `Lexicon.lookup` for heteronyms.
fn lookup(table: &'static HeteronymTable, word: &str, tag: &str) -> Option<Heteronym> {
    let entry = table.get(word).or_else(|| {
        // "Record" at the start of a sentence, but not "RECORD" (could be an acronym)
        let lower = word.to_lowercase();
        let capitalized = lower[..1].to_uppercase() + &lower[1..];
        (word == capitalized).then(|| table.get(&lower)).flatten()
    })?;
    let key = if entry.contains_key(tag) { tag } else { parent_tag(tag) };
    let phonemes = entry.get(key).or_else(|| entry.get("DEFAULT"))?;
    Some(Heteronym { phonemes, entry })
}

fn parent_tag(tag: &str) -> &str {
    if tag.starts_with("VB") {
        "VERB"
    } else if tag.starts_with("NN") {
        "NOUN"
    } else if tag.starts_with("ADV") || tag.starts_with("RB") {
        "ADV"
    } else if tag.starts_with("ADJ") || tag.starts_with("JJ") {
        "ADJ"
    } else {
        tag
    }
}

/// Replaces the phonemes of each heteronym, given each word's span in `ps`
/// (from `align_words`). Punctuation around a word, like the `.` in
/// `objects.`, is kept. A replacement is skipped if the span doesn't look like
/// the word, since that means the alignment is off. Returns the new phonemes
/// and spans.
pub fn fix_heteronyms(
    ps: &str,
    spans: &[(usize, usize)],
    pw: &[String],
    heteronyms: &[Option<Heteronym>],
) -> (String, Vec<(usize, usize)>) {
    let chars: Vec<char> = ps.chars().collect();
    let mut out = String::new();
    let mut out_len = 0; // in chars
    let mut new_spans = Vec::with_capacity(spans.len());
    let mut copied = 0; // chars of `ps` already copied to `out`
    fn push(out: &mut String, s: &[char]) -> usize {
        out.extend(s);
        s.len()
    }

    for (i, &(a, b)) in spans.iter().enumerate() {
        out_len += push(&mut out, &chars[copied..a]);
        let start = out_len;
        let span = &chars[a..b];
        let (lead, core, trail) = split_punctuation(span);
        let replacement = heteronyms[i]
            .as_ref()
            .filter(|h| looks_like(core, &pw[i], h))
            .map(|h| h.phonemes);
        match replacement {
            Some(r) => {
                out_len += push(&mut out, lead);
                out.push_str(r);
                out_len += r.chars().count();
                out_len += push(&mut out, trail);
            }
            None => out_len += push(&mut out, span),
        }
        new_spans.push((start, out_len));
        copied = b;
    }
    push(&mut out, &chars[copied..]);
    (out, new_spans)
}

/// Splits phonemes into leading punctuation, the word, and trailing punctuation.
fn split_punctuation(span: &[char]) -> (&[char], &[char], &[char]) {
    let is_punct = |c: &char| c.is_whitespace() || PUNCTUATION.contains(*c);
    let start = span.iter().position(|c| !is_punct(c)).unwrap_or(span.len());
    let end = span.iter().rposition(|c| !is_punct(c)).map_or(start, |i| i + 1);
    (&span[..start], &span[start..end], &span[end..])
}

/// Whether `core` (from the chunk's phonemes) is close to the word's isolated
/// phonemes `pw` or to one of its known pronunciations, ignoring punctuation
/// and stress. eSpeak may pronounce the word differently in context than on
/// its own, which is the whole point of fixing heteronyms.
fn looks_like(core: &[char], pw: &str, heteronym: &Heteronym) -> bool {
    let core = norm(core);
    let pw: Vec<char> = pw.chars().collect();
    let (_, pw_core, _) = split_punctuation(&pw);
    std::iter::once(pw_core.to_vec())
        .chain(heteronym.entry.values().map(|v| v.chars().collect()))
        .any(|candidate| {
            let candidate = norm(&candidate);
            levenshtein(&core, &candidate) as f64 <= (0.4 * candidate.len() as f64).max(2.0)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::espeak::Espeak;
    use crate::text_segmentation::{align_words, split_words};

    /// Phonemizes `text` as one chunk with heteronyms applied.
    fn phonemize(text: &str, dialect: Dialect) -> String {
        let espeak = Espeak::start().unwrap();
        let voice = dialect.espeak_voice();
        let words = split_words(text);
        let pw: Vec<String> = words.iter().map(|w| espeak.phonemize(&w.surface, voice).unwrap()).collect();
        let ps = espeak.phonemize(text, voice).unwrap();
        let spans = align_words(&pw, &ps).unwrap();
        fix_heteronyms(&ps, &spans, &pw, &find_heteronyms(&words, dialect)).0
    }

    #[test]
    fn tokenize_splits_punctuation_and_clitics() {
        assert_eq!(tokenize("\"Don't").0, ["``", "Do", "n't"]);
        assert_eq!(tokenize("it's").0, ["it", "'s"]);
        assert_eq!(tokenize("objects.\"").0, ["objects", ".", "''"]);
        assert_eq!(tokenize("$19.99,").0, ["$", "19.99", ","]);
        assert_eq!(tokenize("—"), (vec!["—".to_string()], None));
    }

    #[test]
    fn picks_pronunciation_by_part_of_speech() {
        let cases = [
            ("They record every game.", "ɹəkˈɔɹd"),
            ("I bought the record yesterday.", "ɹˈɛkəɹd"),
            ("I read the book yesterday.", "ɹˈɛd"),
            ("I will read it again.", "ɹˈid"),
            ("She lives here.", "lˈɪvz"),
            ("We heard live music.", "lˈIv"),
            ("Please close the door.", "klˈOz"),
            ("The store is close to here.", "klˈOs"),
            ("They present the award.", "pɹizˈɛnt"),
            ("I got a present.", "pɹˈɛzᵊnt"),
        ];
        for (text, expected) in cases {
            let ps = phonemize(text, Dialect::American);
            assert!(ps.contains(expected), "{text:?} -> {ps:?}, expected {expected:?}");
        }
    }

    #[test]
    fn uses_british_pronunciations_for_british_english() {
        let cases = [
            ("They record every game.", "ɹɪkˈɔːd"),
            ("I bought the record yesterday.", "ɹˈɛkɔːd"),
            ("Please close the door.", "klˈQz"),
            ("They present the award.", "pɹɪzˈɛnt"),
        ];
        for (text, expected) in cases {
            let ps = phonemize(text, Dialect::British);
            assert!(ps.contains(expected), "{text:?} -> {ps:?}, expected {expected:?}");
        }
    }
}
