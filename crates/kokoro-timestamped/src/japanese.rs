//! Japanese phonemization: a port of misaki's `ja.JAG2P` in its default
//! `cutlet` mode, which the Python backend uses for Japanese voices.
//! Morphemes and their readings come from vibrato with UniDic 3.1.1 (the
//! Python version uses MeCab, through fugashi, with UniDic 3.1.0).
//!
//! Ported from `misaki/cutlet.py` (Apache-2.0, hexgrad), itself adapted from
//! cutlet (MIT, Copyright (c) 2020 Paul O'Leary McCann); see `NOTICE.md`.
//! Changes from the original:
//! - UniDic's `*` reading counts as no reading, so punctuation is kept.
//! - Parentheses are turned back from `« »`.
//! - NFKC normalization replaces mojimoji's width conversions.
//! - MeCab's character categories are reduced to what cutlet uses.

use std::collections::{HashMap, HashSet};
use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use unicode_normalization::UnicodeNormalization;
use vibrato::dictionary::LexType;
use vibrato::{Dictionary, Tokenizer};

use crate::embedded::decompress_xz;
use crate::error::{Error, Result};
use crate::num2kana;
use crate::text_segmentation::{Word, chunk_japanese};

/// Any error while getting the dictionary; they come from many libraries.
type BoxError = Box<dyn std::error::Error + Send + Sync>;

const DICT_NAME: &str = "unidic-cwj-3_1_1+compact";
const DICT_URL: &str =
    "https://github.com/daac-tools/vibrato/releases/download/v0.5.0/unidic-cwj-3_1_1+compact.tar.xz";

// UniDic feature columns
const POS1: usize = 0;
const PRON: usize = 9;
const KANA: usize = 20;

// MeCab character categories (UniDic's char.def) that cutlet distinguishes
const SYMBOL: u8 = 3;
const HIRAGANA: u8 = 6;
const KATAKANA: u8 = 7;
const OTHER: u8 = 0;

const SUTEGANA: &str = "ゃゅょぁぃぅぇぉ";
const ODORI: &str = "〃々ゝゞヽ";

static JA_WORDS: LazyLock<HashSet<String>> = LazyLock::new(|| {
    // 330 KB instead of 1.9 MB.
    let words = decompress_xz(include_bytes!("../data/ja_words.txt.xz"));
    let words = String::from_utf8(words).expect("Japanese word list is UTF-8");
    words.lines().map(|word| word.trim().to_string()).collect()
});

static TABLE: LazyLock<HashMap<&'static str, &'static str>> =
    LazyLock::new(|| HEPBURN.into_iter().collect());

/// Kana and Japanese punctuation to phonemes: cutlet's `HEPBURN`, generated from misaki.
const HEPBURN: [(&str, &str); 189] = [
    ("ぁ", "a"),
    ("あ", "a"),
    ("ぃ", "i"),
    ("い", "i"),
    ("ぅ", "ɯ"),
    ("う", "ɯ"),
    ("ぇ", "e"),
    ("え", "e"),
    ("ぉ", "o"),
    ("お", "o"),
    ("か", "ka"),
    ("が", "ɡa"),
    ("き", "kʲi"),
    ("ぎ", "ɡʲi"),
    ("く", "kɯ"),
    ("ぐ", "ɡɯ"),
    ("け", "ke"),
    ("げ", "ɡe"),
    ("こ", "ko"),
    ("ご", "ɡo"),
    ("さ", "sa"),
    ("ざ", "ʣa"),
    ("し", "ɕi"),
    ("じ", "ʥi"),
    ("す", "sɨ"),
    ("ず", "zɨ"),
    ("せ", "se"),
    ("ぜ", "ʣe"),
    ("そ", "so"),
    ("ぞ", "ʣo"),
    ("た", "ta"),
    ("だ", "da"),
    ("ち", "ʨi"),
    ("ぢ", "ʥi"),
    ("つ", "ʦɨ"),
    ("づ", "zɨ"),
    ("て", "te"),
    ("で", "de"),
    ("と", "to"),
    ("ど", "do"),
    ("な", "na"),
    ("に", "ɲi"),
    ("ぬ", "nɯ"),
    ("ね", "ne"),
    ("の", "no"),
    ("は", "ha"),
    ("ば", "ba"),
    ("ぱ", "pa"),
    ("ひ", "çi"),
    ("び", "bʲi"),
    ("ぴ", "pʲi"),
    ("ふ", "ɸɯ"),
    ("ぶ", "bɯ"),
    ("ぷ", "pɯ"),
    ("へ", "he"),
    ("べ", "be"),
    ("ぺ", "pe"),
    ("ほ", "ho"),
    ("ぼ", "bo"),
    ("ぽ", "po"),
    ("ま", "ma"),
    ("み", "mʲi"),
    ("む", "mɯ"),
    ("め", "me"),
    ("も", "mo"),
    ("ゃ", "ja"),
    ("や", "ja"),
    ("ゅ", "jɯ"),
    ("ゆ", "jɯ"),
    ("ょ", "jo"),
    ("よ", "jo"),
    ("ら", "ɾa"),
    ("り", "ɾʲi"),
    ("る", "ɾɯ"),
    ("れ", "ɾe"),
    ("ろ", "ɾo"),
    ("ゎ", "βa"),
    ("わ", "βa"),
    ("ゐ", "i"),
    ("ゑ", "e"),
    ("を", "o"),
    ("ゔ", "vɯ"),
    ("ゕ", "ka"),
    ("ゖ", "ke"),
    ("ヷ", "va"),
    ("ヸ", "vʲi"),
    ("ヹ", "ve"),
    ("ヺ", "vo"),
    ("いぇ", "je"),
    ("うぃ", "βi"),
    ("うぇ", "βe"),
    ("うぉ", "βo"),
    ("きぇ", "kʲe"),
    ("きゃ", "kʲa"),
    ("きゅ", "kʲɨ"),
    ("きょ", "kʲo"),
    ("ぎゃ", "ɡʲa"),
    ("ぎゅ", "ɡʲɨ"),
    ("ぎょ", "ɡʲo"),
    ("くぁ", "kᵝa"),
    ("くぃ", "kᵝi"),
    ("くぇ", "kᵝe"),
    ("くぉ", "kᵝo"),
    ("ぐぁ", "ɡᵝa"),
    ("ぐぃ", "ɡᵝi"),
    ("ぐぇ", "ɡᵝe"),
    ("ぐぉ", "ɡᵝo"),
    ("しぇ", "ɕe"),
    ("しゃ", "ɕa"),
    ("しゅ", "ɕɨ"),
    ("しょ", "ɕo"),
    ("じぇ", "ʥe"),
    ("じゃ", "ʥa"),
    ("じゅ", "ʥɨ"),
    ("じょ", "ʥo"),
    ("ちぇ", "ʨe"),
    ("ちゃ", "ʨa"),
    ("ちゅ", "ʨɨ"),
    ("ちょ", "ʨo"),
    ("ぢゃ", "ʥa"),
    ("ぢゅ", "ʥɨ"),
    ("ぢょ", "ʥo"),
    ("つぁ", "ʦa"),
    ("つぃ", "ʦʲi"),
    ("つぇ", "ʦe"),
    ("つぉ", "ʦo"),
    ("てぃ", "tʲi"),
    ("てゅ", "tʲɨ"),
    ("でぃ", "dʲi"),
    ("でゅ", "dʲɨ"),
    ("とぅ", "tɯ"),
    ("どぅ", "dɯ"),
    ("にぇ", "ɲe"),
    ("にゃ", "ɲa"),
    ("にゅ", "ɲɨ"),
    ("にょ", "ɲo"),
    ("ひぇ", "çe"),
    ("ひゃ", "ça"),
    ("ひゅ", "çɨ"),
    ("ひょ", "ço"),
    ("びゃ", "bʲa"),
    ("びゅ", "bʲɨ"),
    ("びょ", "bʲo"),
    ("ぴゃ", "pʲa"),
    ("ぴゅ", "pʲɨ"),
    ("ぴょ", "pʲo"),
    ("ふぁ", "ɸa"),
    ("ふぃ", "ɸʲi"),
    ("ふぇ", "ɸe"),
    ("ふぉ", "ɸo"),
    ("ふゅ", "ɸʲɨ"),
    ("ふょ", "ɸʲo"),
    ("みゃ", "mʲa"),
    ("みゅ", "mʲɨ"),
    ("みょ", "mʲo"),
    ("りゃ", "ɾʲa"),
    ("りゅ", "ɾʲɨ"),
    ("りょ", "ɾʲo"),
    ("ゔぁ", "va"),
    ("ゔぃ", "vʲi"),
    ("ゔぇ", "ve"),
    ("ゔぉ", "vo"),
    ("ゔゅ", "bʲɨ"),
    ("ゔょ", "bʲo"),
    ("。", "."),
    ("、", ","),
    ("？", "?"),
    ("！", "!"),
    ("「", "“"),
    ("」", "”"),
    ("『", "“"),
    ("』", "”"),
    ("：", ":"),
    ("；", ";"),
    ("（", "("),
    ("）", ")"),
    ("《", "("),
    ("》", ")"),
    ("【", "["),
    ("】", "]"),
    ("・", " "),
    ("，", ","),
    ("～", "—"),
    ("〜", "—"),
    ("—", "—"),
    ("«", "“"),
    ("»", "”"),
    ("\u{309A}", ""),
    ("\u{3099}", ""),
];

pub struct Japanese {
    tokenizer: Tokenizer,
}

struct Morpheme {
    surface: String,
    range: std::ops::Range<usize>,
    features: Vec<String>,
    unknown: bool,
}

/// A word as cutlet sees it: its text, its reading in hiragana, and what
/// kind of characters it's made of.
#[derive(Clone)]
struct CutletWord {
    surface: String,
    hira: String,
    char_type: u8,
}

struct Token {
    surface: String,
    /// Whether a space follows.
    space: bool,
}

impl Japanese {
    /// Loads the UniDic dictionary, downloading it on first use.
    pub async fn load() -> Result<Self> {
        let tokenizer = load_tokenizer().await.map_err(Error::JapaneseDictionary)?;
        Ok(Self { tokenizer })
    }


    fn analyze(&self, text: &str) -> Vec<Morpheme> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut worker = self.tokenizer.new_worker();
        worker.reset_sentence(text);
        worker.tokenize();
        (0..worker.num_tokens())
            .map(|i| {
                let token = worker.token(i);
                Morpheme {
                    surface: token.surface().to_string(),
                    range: token.range_byte(),
                    features: parse_csv(token.feature()),
                    unknown: token.lex_type() == LexType::Unknown,
                }
            })
            .collect()
    }

    /// Splits `clean` into word-like chunks for subtitles.
    pub fn split_words(&self, clean: &str) -> Vec<Word> {
        let morphemes = self.analyze(clean);
        let pos: Vec<_> = morphemes
            .iter()
            .map(|m| (m.range.clone(), m.features.get(POS1).map_or("", String::as_str)))
            .collect();
        chunk_japanese(&pos)
            .into_iter()
            .map(|range| Word {
                surface: clean[range.clone()].to_string(),
                range,
            })
            .collect()
    }

    /// Converts `text` to Kokoro phonemes (cutlet's `Cutlet.__call__`).
    pub fn phonemize(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        let text = normalize(text);
        let words = self
            .analyze(&text)
            .into_iter()
            .map(|m| {
                // UniDic gives punctuation the reading `*`. Unlike the Python
                // version, treat that as no reading, so the punctuation itself
                // is kept (`。` becomes `.`) and sentences don't run together.
                let reading = [PRON, KANA]
                    .iter()
                    .filter_map(|&i| m.features.get(i))
                    .find(|f| !f.is_empty() && *f != "*")
                    .map_or(m.surface.as_str(), String::as_str);
                let category = char_category(m.surface.chars().next().unwrap_or(' '));
                CutletWord {
                    hira: kata2hira(reading),
                    char_type: if !m.unknown || category == KATAKANA { HIRAGANA } else { category },
                    surface: m.surface,
                }
            })
            .collect();

        let out: String = romaji_tokens(words)
            .iter()
            .map(|t| format!("{}{}", t.surface, if t.space { " " } else { "" }))
            .collect();
        let ps = out
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .replace('(', "«")
            .replace(')', "»");
        // Unlike the Python version, turn the parentheses back (like misaki's
        // eSpeak G2P does), since Kokoro has no « » and would drop them.
        remove_spaces_around_glottal_stops(&ps)
            .replace('«', "(")
            .replace('»', ")")
    }
}

async fn load_tokenizer() -> Result<Tokenizer, BoxError> {
    let path = dictionary_path()?;
    if !path.exists() {
        download_dictionary(&path).await?;
    }
    tokio::task::spawn_blocking(move || -> Result<Tokenizer, BoxError> {
        let file = BufReader::new(std::fs::File::open(&path)?);
        let decoder = BufReader::new(ruzstd::decoding::StreamingDecoder::new(file)?);
        let dictionary = Dictionary::read(decoder)?;
        // Like MeCab: spaces aren't morphemes, and unknown words are
        // grouped up to 24 characters.
        Ok(Tokenizer::new(dictionary).ignore_space(true)?.max_grouping_len(24))
    })
    .await?
}

fn dictionary_path() -> Result<PathBuf, BoxError> {
    Ok(dirs::cache_dir()
        .ok_or("no cache directory")?
        .join(env!("CARGO_PKG_NAME"))
        .join(DICT_NAME)
        .join("system.dic.zst"))
}

async fn download_dictionary(path: &Path) -> Result<(), BoxError> {
    log::info!("downloading Japanese dictionary from {DICT_URL}");
    let archive = reqwest::get(DICT_URL)
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || -> Result<(), BoxError> {
        let mut tar_bytes = Vec::new();
        lzma_rs::xz_decompress(&mut Cursor::new(&archive[..]), &mut tar_bytes)
            .map_err(|e| format!("cannot decompress {DICT_URL}: {e:?}"))?;
        for entry in tar::Archive::new(Cursor::new(tar_bytes)).entries()? {
            let mut entry = entry?;
            if entry.path()?.ends_with("system.dic.zst") {
                std::fs::create_dir_all(path.parent().unwrap())?;
                // Unpack under another name first, so an interrupted
                // download doesn't leave a broken dictionary behind.
                let partial = path.with_extension("part");
                entry.unpack(&partial)?;
                std::fs::rename(&partial, &path)?;
                return Ok(());
            }
        }
        Err(format!("{DICT_URL} contains no system.dic.zst").into())
    })
    .await?
}

/// Splits a CSV line, allowing quoted fields (UniDic quotes fields with commas).
fn parse_csv(line: &str) -> Vec<String> {
    let mut fields = vec![String::new()];
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                chars.next();
                fields.last_mut().unwrap().push('"');
            }
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(String::new()),
            _ => fields.last_mut().unwrap().push(c),
        }
    }
    fields
}

/// The MeCab character category of `c`, as far as cutlet cares.
fn char_category(c: char) -> u8 {
    const KATAKANA_RANGES: &[(u32, u32)] = &[(0x30A1, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)];
    // Later lines of char.def override earlier ones, so e.g. 々 and 〇 are
    // symbols. Katakana and symbol ranges don't overlap.
    const SYMBOL_RANGES: &[(u32, u32)] = &[
        (0x0021, 0x002F), (0x003A, 0x0040), (0x005B, 0x0060), (0x007B, 0x007E),
        (0x00A1, 0x00BF), (0xFF01, 0xFF0F), (0xFF1A, 0xFF20), (0xFF3B, 0xFF40),
        (0xFF5B, 0xFF65), (0xFFE0, 0xFFEF), (0x2000, 0x206F), (0x20A0, 0x20CF),
        (0x20D0, 0x20FF), (0x2100, 0x214F), (0x2190, 0x21FF), (0x2200, 0x22FF),
        (0x2300, 0x23FF), (0x2460, 0x24FF), (0x2501, 0x257F), (0x2580, 0x259F),
        (0x25A0, 0x25FF), (0x2600, 0x26FE), (0x2700, 0x27BF), (0x27F0, 0x27FF),
        (0x27C0, 0x27EF), (0x2800, 0x28FF), (0x2900, 0x297F), (0x2B00, 0x2BFF),
        (0x2A00, 0x2AFF), (0x3300, 0x33FF), (0x3200, 0x32FE), (0x3000, 0x303F),
        (0xFE30, 0xFE4F), (0xFE50, 0xFE6B),
    ];
    let in_any = |ranges: &[(u32, u32)]| ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&(c as u32)));
    if in_any(KATAKANA_RANGES) {
        KATAKANA
    } else if in_any(SYMBOL_RANGES) {
        SYMBOL
    } else {
        OTHER
    }
}

/// jaconv's `kata2hira`.
fn kata2hira(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            'ァ'..='ヶ' | 'ヽ' | 'ヾ' => char::from_u32(c as u32 - 0x60).unwrap(),
            _ => c,
        })
        .collect()
}

/// cutlet's `_normalize_text`. NFKC also does what mojimoji does there:
/// full-width letters and digits to half-width, half-width katakana to full-width.
fn normalize(text: &str) -> String {
    let is_digit = |c: &char| c.is_ascii_digit() || ('０'..='９').contains(c);
    let chars: Vec<char> = text.chars().collect();
    let mut s = String::new();
    for (i, &c) in chars.iter().enumerate() {
        // a wave dash before a number means a range
        if (c == '〜' || c == '～') && chars.get(i + 1).is_some_and(is_digit) {
            s.push_str("から");
        } else {
            s.push(match c {
                'ㇰ' => 'ク', 'ㇱ' => 'シ', 'ㇲ' => 'ス', 'ㇳ' => 'ト',
                'ㇴ' => 'ヌ', 'ㇵ' => 'ハ', 'ㇶ' => 'ヒ', 'ㇷ' => 'フ',
                'ㇸ' => 'ヘ', 'ㇹ' => 'ホ', 'ㇺ' => 'ム', 'ㇻ' => 'ラ',
                'ㇼ' => 'リ', 'ㇽ' => 'ル', 'ㇾ' => 'レ', 'ㇿ' => 'ロ',
                _ => c,
            });
        }
    }

    // Numbers are read out in hiragana, after a space.
    let mut out = String::new();
    let mut digits = String::new();
    for c in s.nfkc().chain([' ']) {
        if c.is_ascii_digit() {
            digits.push(c);
            continue;
        }
        if !digits.is_empty() {
            out.push(' ');
            out.push_str(&num2kana::convert(&digits));
            digits.clear();
        }
        out.push(c);
    }
    out.pop(); // the extra space added to flush the last number
    out
}

/// cutlet's `_romaji_tokens`.
fn romaji_tokens(words: Vec<CutletWord>) -> Vec<Token> {
    // Merge runs of morphemes that make up a known word, so they're spaced as one.
    let mut groups = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let z = (i + 1..words.len())
            .find(|&z| words[z].char_type != words[i].char_type)
            .unwrap_or(words.len());
        let joined = |j: usize| words[i..j].iter().map(|w| w.surface.as_str()).collect::<String>();
        match (i + 1..=z).rev().find(|&j| JA_WORDS.contains(joined(j).as_str())) {
            None => {
                groups.push(words[i].clone());
                i += 1;
            }
            Some(j) => {
                groups.push(CutletWord {
                    surface: joined(j),
                    hira: words[i..j].iter().map(|w| w.hira.as_str()).collect(),
                    char_type: words[i].char_type,
                });
                i = j;
            }
        }
    }

    let mut out: Vec<Token> = Vec::new();
    for word in &groups {
        let roma = romaji_word(word);
        let surface = word.surface.as_str();
        // Like the Python `in`, these are substring tests, so an empty
        // reading counts as an opening bracket.
        let space = if "「『«".contains(surface) || "([".contains(roma.as_str()) {
            if let Some(prev) = out.last_mut() {
                prev.space = true;
            }
            false
        } else if "」』»".contains(surface) || "]).,?!:".contains(roma.as_str()) {
            if let Some(prev) = out.last_mut() {
                prev.space = false;
            }
            true
        } else {
            roma != " "
        };
        out.push(Token { surface: roma, space });
    }
    // remove any leftover sokuon
    for token in &mut out {
        token.surface = token.surface.replace('っ', "");
    }
    out
}

/// cutlet's `_romaji_word`.
fn romaji_word(word: &CutletWord) -> String {
    if word.surface.is_ascii() {
        return word.surface.clone();
    }
    if word.char_type == SYMBOL {
        return word
            .surface
            .chars()
            .map(|c| {
                let s = c.to_string();
                TABLE.get(s.as_str()).map_or(s, |m| m.to_string())
            })
            .collect();
    }
    if word.char_type != HIRAGANA {
        return String::new();
    }
    let hira: Vec<char> = word.hira.chars().collect();
    (0..hira.len())
        .map(|k| {
            let prev = k.checked_sub(1).map(|p| hira[p]);
            single_mapping(prev, hira[k], hira.get(k + 1).copied())
        })
        .collect()
}

fn table(kana: &[char]) -> Option<&'static str> {
    TABLE.get(kana.iter().collect::<String>().as_str()).copied()
}

/// cutlet's `_get_single_mapping`: the phonemes for kana `kk` between `pk` and `nk`.
fn single_mapping(pk: Option<char>, kk: char, nk: Option<char>) -> String {
    // iteration marks
    if ODORI.contains(kk) {
        return match (kk, pk) {
            ('ゝ' | 'ヽ', Some(p)) => p.to_string(),
            ('ゞ', Some(p)) => add_dakuten(p)
                .and_then(|v| table(&[v]))
                .unwrap_or("")
                .to_string(),
            _ => String::new(),
        };
    }
    // digraphs like きゃ
    if let Some(p) = pk {
        if let Some(m) = table(&[p, kk]) {
            return m.to_string();
        }
    }
    if let Some(n) = nk {
        if table(&[kk, n]).is_some() {
            return String::new();
        }
        if SUTEGANA.contains(n) {
            if kk == 'っ' {
                return String::new();
            }
            let mut base = table(&[kk]).unwrap_or("").to_string();
            base.pop();
            return base + table(&[n]).unwrap_or("");
        }
    }
    if SUTEGANA.contains(kk) {
        return String::new();
    }
    match kk {
        'ー' => "ː".to_string(),
        'っ' => "ʔ".to_string(),
        // ん assimilates to the following sound
        'ん' => {
            let next = nk.and_then(|n| table(&[n])).unwrap_or("");
            if next.starts_with(['m', 'p', 'b']) {
                "m"
            } else if next.starts_with(['k', 'ɡ']) {
                "ŋ"
            } else if next.starts_with(['ɲ', 'ʨ', 'ʥ']) {
                "ɲ"
            } else if next.starts_with(['n', 't', 'd', 'ɾ', 'z']) {
                "n"
            } else {
                "ɴ"
            }
            .to_string()
        }
        _ => table(&[kk]).unwrap_or("").to_string(),
    }
}

fn add_dakuten(kana: char) -> Option<char> {
    let plain = "かきくけこさしすせそたちつてとはひふへほ";
    let voiced = "がぎぐげござじずぜぞだぢづでどばびぶべぼ";
    let i = plain.chars().position(|c| c == kana)?;
    voiced.chars().nth(i)
}

/// Removes a space before `ʔ` (unless after punctuation) and after `ʔ`
/// (unless before a quote).
fn remove_spaces_around_glottal_stops(ps: &str) -> String {
    let chars: Vec<char> = ps.chars().collect();
    chars
        .iter()
        .enumerate()
        .filter(|&(i, &c)| {
            if c != ' ' {
                return true;
            }
            let prev = i.checked_sub(1).map(|p| chars[p]);
            let next = chars.get(i + 1).copied();
            let before_stop = next == Some('ʔ') && !prev.is_some_and(|p| "!\",.:;?»—…”".contains(p));
            let after_stop = prev == Some('ʔ') && !next.is_some_and(|n| "\"«“".contains(n));
            !(before_stop || after_stop)
        })
        .map(|(_, &c)| c)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values are the Python backend's output for the same text,
    /// except that punctuation is kept. The first run downloads the dictionary.
    #[tokio::test]
    async fn matches_python_backend_but_keeps_punctuation() {
        let japanese = Japanese::load().await.unwrap();
        let cases = [
            ("こんにちは、元気ですか？", "koɲɲiʨiβa, ɡeŋkʲi desɨ ka?"),
            ("「ちょっと待って！」と彼は言った。", "“ʨoʔto maʔte!” to kaɾe βa iʔta."),
            ("価格は1,980円で、20%オフです。", "kakakɯ βa iʨi, kʲɨːçakɯ haʨiʥɨː eɴ de, ɲiʥɨː % oɸɯ desɨ."),
            ("新幹線で大阪まで行って、たこ焼きを食べたい。", "ɕiŋkaɴseɴ de oːsaka made iʔte, takojakʲi o tabe tai."),
            ("ユーザーがYouTubeで猫の動画を見ています。", "jɯːʣaː ɡa You Tube de neko no doːɡa o mʲite imasɨ."),
            ("（注）この情報は古いかもしれません。", "(ʨɨː) kono ʥoːhoː βa ɸɯɾɯi ka mo ɕiɾe maseɴ."),
            ("ああ、そうなんだ。ふーん。", "aː, soː nanda. ɸɯːɴ."),
        ];
        for (text, expected) in cases {
            assert_eq!(japanese.phonemize(text).trim(), expected, "for {text:?}");
        }

        let words: Vec<String> = japanese
            .split_words("今日は東京の新しいカフェでコーヒーを飲みました。")
            .into_iter()
            .map(|w| w.surface)
            .collect();
        assert_eq!(words, ["今日は", "東京の", "新しい", "カフェで", "コーヒーを", "飲みました。"]);
    }
}
