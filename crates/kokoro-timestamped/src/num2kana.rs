//! Reads numbers in Japanese hiragana: a port of misaki's `num2kana.Convert`
//! (Apache-2.0, hexgrad), itself copied from Convert-Numbers-to-Japanese
//! (MIT, Copyright (c) 2018 David Wilson); see `NOTICE.md`. Only whole
//! numbers in hiragana are ported, not decimals, kanji or romaji.

fn kana(key: &str) -> &'static str {
    match key {
        "0" => "ゼロ",
        "1" => "いち",
        "2" => "に",
        "3" => "さん",
        "4" => "よん",
        "5" => "ご",
        "6" => "ろく",
        "7" => "なな",
        "8" => "はち",
        "9" => "きゅう",
        "10" => "じゅう",
        "100" => "ひゃく",
        "1000" => "せん",
        "10000" => "まん",
        "100000000" => "おく",
        "300" => "さんびゃく",
        "600" => "ろっぴゃく",
        "800" => "はっぴゃく",
        "3000" => "さんぜん",
        "8000" => "はっせん",
        "01000" => "いっせん",
        _ => unreachable!("no reading for {key}"),
    }
}

/// Reads a string of ASCII digits, e.g. `"123"` -> `"ひゃくにじゅうさん"`.
pub fn convert(digits: &str) -> String {
    let digits = digits.replace(',', "");
    if digits.len() > 9 {
        // Like the Python version, which returns this message as the result.
        return "Number length too long, choose less than 10 digits".to_string();
    }
    let digits = match digits.trim_start_matches('0') {
        "" => "0",
        trimmed => trimmed,
    };
    let result = match digits.len() {
        1 => kana(digits).to_string(),
        2 => len_two(digits),
        3 => len_three(digits),
        4 => len_four(digits, true),
        _ => len_x(digits),
    };
    result.replace(' ', "")
}

fn len_two(n: &str) -> String {
    let (first, second) = (&n[..1], &n[1..2]);
    if first == "0" {
        kana(second).to_string()
    } else if n == "10" {
        kana("10").to_string()
    } else if first == "1" {
        format!("{} {}", kana("10"), kana(second))
    } else if second == "0" {
        format!("{} {}", kana(first), kana("10"))
    } else {
        format!("{} {} {}", kana(first), kana("10"), kana(second))
    }
}

fn len_three(n: &str) -> String {
    let mut parts = Vec::new();
    match &n[..1] {
        "1" => parts.push(kana("100").to_string()),
        d @ ("3" | "6" | "8") => parts.push(kana(&format!("{d}00")).to_string()),
        d => parts.extend([kana(d).to_string(), kana("100").to_string()]),
    }
    if &n[1..] != "00" {
        if &n[1..2] == "0" {
            parts.push(kana(&n[2..3]).to_string());
        } else {
            parts.push(len_two(&n[1..]));
        }
    }
    parts.join(" ")
}

fn len_four(n: &str, stand_alone: bool) -> String {
    if n == "0000" {
        return String::new();
    }
    let n = n.trim_start_matches('0');
    match n.len() {
        1 => return kana(n).to_string(),
        2 => return len_two(n),
        3 => return len_three(n),
        _ => {}
    }
    let mut parts = Vec::new();
    match &n[..1] {
        // 1000 on its own is せん, but いっせん inside a larger number
        "1" if stand_alone => parts.push(kana("1000").to_string()),
        "1" => parts.push(kana("01000").to_string()),
        d @ ("3" | "8") => parts.push(kana(&format!("{d}000")).to_string()),
        d => parts.extend([kana(d).to_string(), kana("1000").to_string()]),
    }
    if &n[1..] != "000" {
        if &n[1..2] == "0" {
            parts.push(len_two(&n[2..]));
        } else {
            parts.push(len_three(&n[1..]));
        }
    }
    parts.join(" ")
}

/// 5 to 9 digits.
fn len_x(n: &str) -> String {
    let (high, low) = n.split_at(n.len() - 4);
    let mut parts = Vec::new();
    match high.len() {
        1 => parts.extend([kana(high).to_string(), kana("10000").to_string()]),
        2 => parts.extend([len_two(high), kana("10000").to_string()]),
        3 => parts.extend([len_three(high), kana("10000").to_string()]),
        4 => parts.extend([len_four(high, false), kana("10000").to_string()]),
        _ => {
            parts.extend([kana(&n[..1]).to_string(), kana("100000000").to_string()]);
            parts.push(len_four(&n[1..5], false));
            if &n[1..5] != "0000" {
                parts.push(kana("10000").to_string());
            }
        }
    }
    parts.push(len_four(low, false));
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::convert;

    #[test]
    fn reads_numbers() {
        assert_eq!(convert("3"), "さん");
        assert_eq!(convert("20"), "にじゅう");
        assert_eq!(convert("1980"), "せんきゅうひゃくはちじゅう");
        assert_eq!(convert("2024"), "にせんにじゅうよん");
        assert_eq!(convert("12345"), "いちまんにせんさんびゃくよんじゅうご");
        assert_eq!(convert("007"), "なな");
        assert_eq!(convert("10000"), "いちまん");
        assert_eq!(convert("100000000"), "いちおく");
        assert_eq!(convert("8000"), "はっせん");
        assert_eq!(convert("105"), "ひゃくご");
        assert_eq!(convert("1050"), "せんごじゅう");
        assert_eq!(
            convert("999999999"),
            "きゅうおくきゅうせんきゅうひゃくきゅうじゅうきゅうまんきゅうせんきゅうひゃくきゅうじゅうきゅう"
        );
    }
}
