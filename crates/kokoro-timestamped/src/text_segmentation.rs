//! Port of the Python `text_segmentation.py`: strips `{tags}`, splits text into
//! words, packs words into chunks, and aligns words with the phonemes of the
//! whole chunk so each word can be given a start and end time.

use std::ops::Range;

const SENTENCE_END: [char; 8] = ['.', '?', '!', '。', '！', '？', '।', '॥'];
const CLOSERS: [char; 8] = ['"', '\'', '”', '』', '」', ')', ']', '}'];

/// Phoneme marks that isolated words keep but whole sentences often drop;
/// ignored when scoring the alignment so they don't dominate the edit distance.
const NORM_DROP: [char; 3] = ['ˈ', 'ˌ', 'ː'];

pub struct Word {
    pub surface: String,
    /// Byte range in the clean text.
    pub range: Range<usize>,
}

/// Removes every `{tag}` from `text` and collapses whitespace. Each tag is
/// returned with its byte offset in the clean text.
///
/// The Python version records the number of whitespace-separated words
/// before each tag instead, which puts every tag at the start of Japanese
/// text (no spaces); `tag_word_indices` gives the same result for languages
/// with spaces.
pub fn strip_tags(text: &str) -> (String, Vec<(usize, String)>) {
    let collapse = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut raw = String::new();
    let mut tags = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('{') {
        let Some(len) = rest[open + 1..].find('}') else {
            break;
        };
        raw.push_str(&rest[..open]);
        let name = &rest[open + 1..open + 1 + len];
        // The collapsed text so far is a prefix of the final clean text.
        tags.push((collapse(&raw).len(), name.to_string()));
        rest = &rest[open + 1 + len + 1..];
    }
    raw.push_str(rest);

    (collapse(&raw), tags)
}

/// Converts tag offsets from `strip_tags` into the index of the word each tag
/// fires on (`words.len()` means "at the end").
pub fn tag_word_indices(words: &[Word], tags: Vec<(usize, String)>) -> Vec<(usize, String)> {
    tags.into_iter()
        .map(|(offset, name)| {
            let index = words.iter().take_while(|w| w.range.start < offset).count();
            (index, name)
        })
        .collect()
}

/// Splits `clean` into whitespace-separated words.
pub fn split_words(clean: &str) -> Vec<Word> {
    let mut words = Vec::new();
    let mut start = None;
    for (i, c) in clean.char_indices().chain([(clean.len(), ' ')]) {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                words.push(Word {
                    surface: clean[s..i].to_string(),
                    range: s..i,
                });
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    words
}

/// Groups word indices into `(lo, hi)` chunks whose combined isolated phoneme
/// length stays under `budget`, preferring to end a chunk right after a
/// sentence-final word. `pw[i]` is word `i` phonemized in isolation. A single
/// over-budget word becomes its own chunk.
pub fn pack_words(words: &[Word], pw: &[String], budget: usize) -> Vec<(usize, usize)> {
    let size_of = |i: usize| pw[i].chars().count() + 1;
    let mut result = Vec::new();

    let mut lo = 0;
    let mut size = 0;
    // exclusive index of the last sentence boundary in the current run
    let mut brk: Option<usize> = None;

    for i in 0..words.len() {
        let w_size = size_of(i);

        if size > 0 && size + w_size > budget && i > lo {
            let hi = match brk {
                Some(b) if b > lo => b,
                _ => i,
            };

            result.push((lo, hi));

            lo = hi;
            size = (lo..=i).map(size_of).sum();
            brk = None;
        } else {
            size += w_size;
        }

        if words[i]
            .surface
            .trim_end_matches(CLOSERS)
            .ends_with(SENTENCE_END)
        {
            brk = Some(i + 1);
        }
    }

    if lo < words.len() {
        result.push((lo, words.len()));
    }

    result
}

pub fn levenshtein(a: &[char], b: &[char]) -> usize {
    if a == b {
        return 0;
    }
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            cur.push(
                (prev[j + 1] + 1)
                    .min(cur[j] + 1)
                    .min(prev[j] + usize::from(ca != cb)),
            );
        }
        prev = cur;
    }
    prev[b.len()]
}

pub fn norm(s: &[char]) -> Vec<char> {
    s.iter()
        .filter(|c| !NORM_DROP.contains(c))
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Python's `round()`, which rounds halves to even.
fn round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Partitions `ps` into `pw.len()` consecutive spans matching `pw`.
///
/// `pw[k]` is word `k` phonemized in isolation; `ps` is the whole chunk
/// phonemized together (the string Kokoro synthesizes). Walks a cursor along
/// `ps`, picking the local end that best matches each word's isolated
/// phonemes. The search window is anchored between the cursor and a global
/// proportional estimate, so cross-word effects in `ps` (linking consonants,
/// elisions) that knock the cursor off can't strand it. Returns half-open
/// spans in characters (trimmed of surrounding spaces), or `None` only if the
/// total mismatch is so large the whole pairing must be wrong.
pub fn align_words(pw: &[String], ps: &str) -> Option<Vec<(usize, usize)>> {
    const WINDOW: i64 = 22;

    let ps: Vec<char> = ps.chars().collect();
    let n = pw.len();
    let len = ps.len();
    if n == 0 || len == 0 {
        return None;
    }
    if n == 1 {
        return Some(vec![(0, len)]);
    }

    let norm_pw: Vec<Vec<char>> = pw
        .iter()
        .map(|w| norm(&w.chars().collect::<Vec<_>>()))
        .collect();
    let total_norm = norm_pw.iter().map(Vec::len).sum::<usize>().max(1);
    // ps runs longer than isolated words (stress/length marks, liaison)
    let ratio = len as f64 / total_norm as f64;

    let mut spans = Vec::with_capacity(n);
    let mut cursor = 0;
    let mut consumed = 0; // sum of norm_pw lengths for words already placed
    let mut total_cost = 0.0;
    for (k, target) in norm_pw.iter().enumerate() {
        while cursor < len && ps[cursor] == ' ' {
            cursor += 1;
        }
        let a = cursor;
        let remaining = (n - 1 - k) as i64;
        consumed += target.len();

        let b = if k == n - 1 {
            len
        } else {
            let g_end = consumed as f64 * ratio; // global estimate of word end
            let local_end = a as i64 + round(target.len() as f64 * ratio).max(1);
            let lo = (a as i64 + 1).max(round((local_end as f64).min(g_end)) - WINDOW);
            let hi = (len as i64 - remaining).min(round((local_end as f64).max(g_end)) + WINDOW);
            let hi = hi.min(lo + 120);
            let b = if hi <= lo {
                (len as i64 - remaining).min((a as i64 + 1).max(local_end))
            } else {
                let mut b = lo;
                let mut best = f64::INFINITY;
                for cand in lo..=hi {
                    let c = cand as usize;
                    let mut cost = levenshtein(&norm(&ps[a..c]), target) as f64;
                    if c < len && ps[c] != ' ' && ps[c - 1] != ' ' {
                        cost += 0.5; // prefer cutting on a word boundary
                    }
                    if cost < best {
                        best = cost;
                        b = cand;
                    }
                }
                // A poor local match must not strand the cursor: snap toward
                // the global estimate instead of trusting the bad match.
                if best > (0.6 * target.len() as f64).max(2.0) {
                    b = hi.min(lo.max(round(g_end)));
                }
                total_cost += best;
                b
            };
            // Only reachable with more words than phonemes.
            b.max(0) as usize
        };

        let mut end = b;
        while end > a && ps[end - 1] == ' ' {
            end -= 1;
        }
        spans.push((a, a.max(end)));
        cursor = b;
    }

    if total_cost > 0.6 * len as f64 {
        return None;
    }
    Some(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag_indices(text: &str, words: impl Fn(&str) -> Vec<Word>) -> Vec<(usize, String)> {
        let (clean, tags) = strip_tags(text);
        tag_word_indices(&words(&clean), tags)
    }

    #[test]
    fn tags_fire_on_the_following_word() {
        let expected = |v: &[(usize, &str)]| -> Vec<(usize, String)> {
            v.iter().map(|&(i, n)| (i, n.to_string())).collect()
        };
        assert_eq!(
            tag_indices("{a} Hello  {b}world {c}", split_words),
            expected(&[(0, "a"), (1, "b"), (2, "c")]),
        );
        // A tag inside a word counts it as before the tag, like Python.
        assert_eq!(tag_indices("foo{t}bar", split_words), expected(&[(1, "t")]));

        // Japanese words aren't separated by spaces.
        let japanese_words = |clean: &str| {
            let split = clean.find('元').unwrap();
            vec![
                Word { surface: clean[..split].to_string(), range: 0..split },
                Word { surface: clean[split..].to_string(), range: split..clean.len() },
            ]
        };
        assert_eq!(
            tag_indices("{a}こんにちは！{b}元気？{c}", japanese_words),
            expected(&[(0, "a"), (1, "b"), (2, "c")]),
        );
    }
}
