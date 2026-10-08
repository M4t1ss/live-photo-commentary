//! Phonemization with eSpeak NG, reproducing what the Python backend gets
//! from misaki's `EspeakG2P` (which goes through phonemizer).
//!
//! The `E2M` table and conversion are from misaki's `EspeakG2P` (Apache-2.0,
//! hexgrad). `MARKS`, `split_marks` and putting punctuation back in
//! `phonemize` are modelled on phonemizer's `Punctuation` class
//! (GPL-3.0-or-later); see `NOTICE.md`. Changes: phonemes are requested
//! `_`-separated rather than tied, and the danda (`।`, `॥`) is kept as
//! punctuation.

use std::sync::{Mutex, PoisonError};

use espeak_rs::ESpeakError;

type Result<T> = std::result::Result<T, ESpeakError>;

// espeak-rs-sys uses Windows registry functions without linking advapi32.
#[cfg(windows)]
#[link(name = "advapi32")]
unsafe extern "C" {}

/// eSpeak keeps global state, so only one call may use it at a time. The
/// program is single-threaded, but tests run in parallel.
static ESPEAK_LOCK: Mutex<()> = Mutex::new(());

/// Punctuation kept as-is rather than sent to eSpeak, which would drop it
/// (phonemizer's default marks, plus the Devanagari danda `।` and `॥`).
const MARKS: &str = ";:,.!?¡¿—…\"«»“”(){}[]।॥";

/// Multi-letter eSpeak phonemes that Kokoro writes as a single symbol
/// (misaki's `EspeakG2P.e2m`).
const E2M: [(&str, &str); 11] = [
    ("aɪ", "I"), ("aʊ", "W"),
    ("dz", "ʣ"), ("dʒ", "ʤ"),
    ("eɪ", "A"),
    ("oʊ", "O"), ("əʊ", "Q"),
    ("ss", "S"),
    ("ts", "ʦ"), ("tʃ", "ʧ"),
    ("ɔɪ", "Y"),
];

/// Converts `text` to Kokoro phonemes with the eSpeak voice `voice` (e.g.
/// `en-us`), keeping punctuation in place.
pub fn phonemize(text: &str, voice: &str) -> Result<String> {
    let text = text.replace('«', "“").replace('»', "”");
    let mut out = String::new();
    for (is_mark, piece) in split_marks(&text) {
        if is_mark {
            // Unlike the Python version, which leaves the danda to eSpeak:
            // eSpeak drops it, and the sentences around it run together.
            // Kokoro has no danda, so it becomes a full stop.
            out.push_str(&piece.replace(['।', '॥'], "."));
        } else {
            out.push_str(&phonemize_words(piece, voice)?);
        }
    }
    Ok(out.replace('-', "").trim().to_string())
}

/// Splits `text` into alternating text and punctuation pieces, `(is_mark, piece)`.
/// A punctuation piece is a run of marks and whitespace containing at least
/// one mark, so the spacing around the marks is kept too.
fn split_marks(text: &str) -> Vec<(bool, &str)> {
    let is_sep = |c: char| c.is_whitespace() || MARKS.contains(c);
    let mut pieces = Vec::new();
    let mut text_start = 0;
    let mut i = 0;
    while i < text.len() {
        let c = text[i..].chars().next().unwrap();
        if !is_sep(c) {
            i += c.len_utf8();
            continue;
        }
        let run_end = text[i..]
            .find(|c: char| !is_sep(c))
            .map_or(text.len(), |n| i + n);
        let run = &text[i..run_end];
        if run.contains(|c: char| MARKS.contains(c)) {
            if text_start < i {
                pieces.push((false, &text[text_start..i]));
            }
            pieces.push((true, run));
            text_start = run_end;
        }
        i = run_end;
    }
    if text_start < text.len() {
        pieces.push((false, &text[text_start..]));
    }
    pieces
}

/// Phonemizes text without punctuation. Phonemes come back separated by `_`,
/// which keeps multi-letter phonemes like `oʊ` together so they can be mapped.
fn phonemize_words(text: &str, voice: &str) -> Result<String> {
    let sentences = {
        let _guard = ESPEAK_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        espeak_rs::text_to_phonemes(text, voice, Some('_'))?
    };
    let words: Vec<String> = sentences
        .iter()
        .flat_map(|s| s.split_whitespace())
        .map(|word| word.split('_').map(map_phoneme).collect())
        .collect();
    Ok(words.join(" "))
}

fn map_phoneme(phoneme: &str) -> String {
    let body = phoneme.trim_start_matches(['ˈ', 'ˌ']);
    let stress = &phoneme[..phoneme.len() - body.len()];
    match E2M.iter().find(|(espeak, _)| *espeak == body) {
        Some((_, kokoro)) => format!("{stress}{kokoro}"),
        None => phoneme.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::phonemize;

    /// Expected values are the Python backend's output for the same text.
    #[test]
    fn matches_python_backend() {
        let cases = [
            (
                "Hello, world! I have 123 apples.",
                "həlˈO, wˈɜːld! I hæv wˈʌnhˈʌndɹɪd twˈɛnti θɹˈiː ˈæpəlz.",
            ),
            (
                "It's 3:45 PM and the price is $19.99, roughly 20% off.",
                "ɪts θɹˈiː:fˈɔːɹɾi fˈIv pˌiːˈɛm ænd ðə pɹˈIs ɪz dˈɑːlɚ nˈIntiːn.nˈInti nˈIn, ɹˈʌfli twˈɛnti pɚsˈɛnt ˈɔf.",
            ),
            (
                "Dr. Smith said: \"Don't panic!\" — then left.",
                "dˈɑːktɚ. smˈɪθ sˈɛd: \"dˈOnt pˈænɪk!\" — ðˈɛn lˈɛft.",
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(phonemize(text, "en-us").unwrap(), expected, "for {text:?}");
        }
    }

    #[test]
    fn keeps_hindi_sentence_breaks() {
        // The danda becomes a full stop instead of joining the sentences.
        let first = phonemize("नमस्ते", "hi").unwrap();
        let second = phonemize("आप कैसे हैं?", "hi").unwrap();
        assert_eq!(
            phonemize("नमस्ते। आप कैसे हैं?", "hi").unwrap(),
            format!("{first}. {second}")
        );
    }
}
