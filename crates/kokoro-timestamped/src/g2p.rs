//! Chooses grapheme-to-phoneme conversion by language: misaki's Japanese
//! G2P for Japanese, eSpeak for everything else, like the Python backend.

use crate::english::{self, Dialect, Heteronym};
use crate::error::{Error, Result};
use crate::espeak::Espeak;
use crate::japanese::Japanese;
use crate::text_segmentation::{Word, split_words};

pub enum G2p {
    /// eSpeak plus fixing heteronyms (see `english`).
    English(Dialect, Espeak),
    /// Any other eSpeak voice, by eSpeak's name for it, like `fr`.
    Espeak(&'static str, Espeak),
    Japanese(Japanese),
}

/// The language of a Kokoro voice, from the first letter of its name
/// (`af_heart` is American English), as in the Python `VOICE_LANGS`.
pub fn voice_language(voice: &str) -> Result<&'static str> {
    Ok(match voice.chars().next() {
        Some('a') => "en-us",
        Some('b') => "en-gb",
        Some('j') => "ja",
        Some('z') => "zh",
        Some('e') => "es",
        Some('f') => "fr-fr",
        Some('h') => "hi",
        Some('i') => "it",
        Some('p') => "pt-br",
        _ => return Err(Error::UnknownVoiceLanguage(voice.to_string())),
    })
}

impl G2p {
    pub async fn new(language: &str) -> Result<Self> {
        // eSpeak looks voices up by its own names, which don't always match
        // the language codes.
        let espeak_voice = match language {
            "ja" => return Ok(Self::Japanese(Japanese::load().await?)),
            "en-us" => return Ok(Self::English(Dialect::American, Espeak::start()?)),
            "en-gb" => return Ok(Self::English(Dialect::British, Espeak::start()?)),
            "es" => "es",
            "fr-fr" => "fr",
            "hi" => "hi",
            "it" => "it",
            "pt-br" => "pt-br",
            // Mandarin (zh) would need misaki's Chinese G2P.
            _ => return Err(Error::UnsupportedLanguage(language.to_string())),
        };
        Ok(Self::Espeak(espeak_voice, Espeak::start()?))
    }

    pub fn phonemize(&self, text: &str) -> Result<String> {
        match self {
            Self::English(dialect, espeak) => espeak.phonemize(text, dialect.espeak_voice()),
            Self::Espeak(voice, espeak) => espeak.phonemize(text, voice),
            Self::Japanese(japanese) => Ok(japanese.phonemize(text).trim().to_string()),
        }
    }

    /// Splits `clean` into words for subtitles and alignment.
    pub fn split_words(&self, clean: &str) -> Vec<Word> {
        match self {
            Self::English(..) | Self::Espeak(..) => split_words(clean),
            Self::Japanese(japanese) => japanese.split_words(clean),
        }
    }

    /// For each word, a better pronunciation than the phonemizer's, if known.
    /// Apply them with `english::fix_heteronyms`.
    pub fn heteronyms(&self, words: &[Word]) -> Vec<Option<Heteronym>> {
        match self {
            Self::English(dialect, _) => english::find_heteronyms(words, *dialect),
            Self::Espeak(..) | Self::Japanese(_) => words.iter().map(|_| None).collect(),
        }
    }
}
