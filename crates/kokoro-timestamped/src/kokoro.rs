use std::collections::HashMap;
use std::sync::LazyLock;
use std::path::PathBuf;
use ort::session::Session;
use hf_hub::HFClient;
use ndarray::{Array3, Axis};
use ort::value::Tensor;
use futures::TryStreamExt;
use hf_hub::HFRepository;
use hf_hub::repository::RepoTreeEntry;
use hf_hub::repository::repo_type::RepoTypeModel;
use crate::english::{self, Heteronym};
use crate::error::{Error, Result};
use crate::g2p::{G2p, voice_language};
use crate::text_segmentation::{Word, align_words, pack_words, strip_tags, tag_word_indices};

type ModelRepo = HFRepository<RepoTypeModel>;

type VoiceTensor = Array3<f32>;

/// One voice in a [`Voice::Blend`].
pub struct VoiceBlendPart {
    pub name: String,
    pub weight: f32,
}

/// A Kokoro voice. The first letter of a voice name gives its language, e.g.
/// `af_heart` (American English) or `jf_alpha` (Japanese); see the
/// [voice list](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
pub enum Voice {
    Name(String),
    /// A weighted sum of voices, e.g. 70% `af_heart` and 30% `af_bella`.
    Blend(Vec<VoiceBlendPart>),
    /// A voice tensor of shape `(frames, 1, 256)`, as in the voice files.
    Array(VoiceTensor),
}

/// Raw model output. `starts` has one entry per phoneme character plus one:
/// `starts[0]` is when the first phoneme begins (the leading silence), and
/// `starts[i + 1]` is when phoneme `i` ends. Times are in seconds.
struct ModelOutput {
    audio: Vec<f32>,
    starts: Vec<f64>,
}

/// When a word is spoken. Times are in seconds from the start of the chunk.
pub struct WordTiming {
    pub text: String,
    /// Character (not byte) offsets into the chunk text, like the Python version.
    pub char_start: usize,
    pub char_end: usize,
    pub start: f64,
    pub end: f64,
}

/// One synthesized chunk, the equivalent of the Python `_synth_chunk` tuple.
/// Times are in seconds from the start of the chunk's audio.
pub struct Chunk {
    /// Mono samples at [`KokoroSynthesizer::SAMPLE_RATE`].
    pub audio: Vec<f32>,
    /// The part of the input text this chunk speaks, without tags.
    pub text: String,
    /// Each phoneme character with the time it ends.
    pub phonemes: Vec<(char, f64)>,
    /// `None` if the words could not be aligned with the phonemes.
    pub words: Option<Vec<WordTiming>>,
    /// Each `{tag}` name with the time it fires.
    pub tags: Vec<(String, f64)>,
}

/// Synthesizes speech with Kokoro, through ONNX Runtime.
pub struct KokoroSynthesizer {
    session: ort::session::Session,
    voice: VoiceTensor,
    g2p: G2p,
    speed: f32,
}

/// Phoneme characters to model token IDs, from Kokoro-82M's `config.json`
/// (Apache-2.0, hexgrad).
pub(crate) static VOCAB: LazyLock<HashMap<char, i64>> = LazyLock::new(|| {
    HashMap::from([
        (';', 1), (':', 2), (',', 3), ('.', 4), ('!', 5),
        ('?', 6), ('—', 9), ('…', 10), ('"', 11), ('(', 12),
        (')', 13), ('“', 14), ('”', 15), (' ', 16), ('\u{0303}', 17),
        ('ʣ', 18), ('ʥ', 19), ('ʦ', 20), ('ʨ', 21), ('ᵝ', 22),
        ('\u{AB67}', 23), ('A', 24), ('I', 25), ('O', 31), ('Q', 33),
        ('S', 35), ('T', 36), ('W', 39), ('Y', 41), ('ᵊ', 42),
        ('a', 43), ('b', 44), ('c', 45), ('d', 46), ('e', 47),
        ('f', 48), ('h', 50), ('i', 51), ('j', 52), ('k', 53),
        ('l', 54), ('m', 55), ('n', 56), ('o', 57), ('p', 58),
        ('q', 59), ('r', 60), ('s', 61), ('t', 62), ('u', 63),
        ('v', 64), ('w', 65), ('x', 66), ('y', 67), ('z', 68),
        ('ɑ', 69), ('ɐ', 70), ('ɒ', 71), ('æ', 72), ('β', 75),
        ('ɔ', 76), ('ɕ', 77), ('ç', 78), ('ɖ', 80), ('ð', 81),
        ('ʤ', 82), ('ə', 83), ('ɚ', 85), ('ɛ', 86), ('ɜ', 87),
        ('ɟ', 90), ('ɡ', 92), ('ɥ', 99), ('ɨ', 101), ('ɪ', 102),
        ('ʝ', 103), ('ɯ', 110), ('ɰ', 111), ('ŋ', 112), ('ɳ', 113),
        ('ɲ', 114), ('ɴ', 115), ('ø', 116), ('ɸ', 118), ('θ', 119),
        ('œ', 120), ('ɹ', 123), ('ɾ', 125), ('ɻ', 126), ('ʁ', 128),
        ('ɽ', 129), ('ʂ', 130), ('ʃ', 131), ('ʈ', 132), ('ʧ', 133),
        ('ʊ', 135), ('ʋ', 136), ('ʌ', 138), ('ɣ', 139), ('ɤ', 140),
        ('χ', 142), ('ʎ', 143), ('ʒ', 147), ('ʔ', 148), ('ˈ', 156),
        ('ˌ', 157), ('ː', 158), ('ʰ', 162), ('ʲ', 164), ('↓', 169),
        ('→', 171), ('↗', 172), ('↘', 173), ('ᵻ', 177),
    ])
});

impl KokoroSynthesizer {
    const MODEL_REPO_ORG: &str = "onnx-community";
    const MODEL_REPO_NAME: &str = "Kokoro-82M-v1.0-ONNX-timestamped";
    /// Samples per second of [`Chunk::audio`].
    pub const SAMPLE_RATE: usize = 24000;
    /// The model accepts at most 512 input IDs, two of which are the 0 padding
    /// added at the start and end.
    const MAX_PHONEMES: usize = 512 - 2;
    /// Chunks are packed using each word's phonemes in isolation, which can
    /// differ in length from the chunk phonemized as a whole, so leave a margin.
    const CHUNK_BUDGET: usize = Self::MAX_PHONEMES - 10;

    async fn load_voice(repo: &ModelRepo, name: &str) -> Result<VoiceTensor> {
        let voice_path =
            Self::download_file(repo, &format!("voices/{name}.bin")).await?;
        let bytes = tokio::fs::read(&voice_path).await?;

        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(chunk.try_into().unwrap())
            })
        .collect();

        if bytes.len() % 4 != 0 || !data.len().is_multiple_of(256) {
            return Err(Error::InvalidVoice { name: name.to_string() });
        }

        let frames = data.len() / 256;
        Ok(Array3::from_shape_vec((frames, 1, 256), data)
            .expect("the length should be a multiple of 256"))
    }

    async fn load_blend(repo: &ModelRepo, parts: Vec<VoiceBlendPart>) -> Result<VoiceTensor> {
        let voices = futures::future::try_join_all(
            parts.iter().map(|part| async move {
                let mut voice = Self::load_voice(repo, &part.name).await?;
                voice *= part.weight;
                Ok::<_, Error>(voice)
            })
        ).await?;

        let mut blend: Option<VoiceTensor> = None;
        for voice in voices {
            match &mut blend {
                Some(accum) => *accum += &voice,
                None => blend = Some(voice),
            }
        }
        blend.ok_or(Error::EmptyBlend)
    }

    async fn get_voice(repo: &ModelRepo, voice: Voice) -> Result<VoiceTensor> {
        let loaded_voice = match voice {
            Voice::Array(v) => v,
            Voice::Name(name) => {
                Self::load_voice(&repo, &name).await?
            },
            Voice::Blend(parts) => {
                Self::load_blend(&repo, parts).await?
            },
        };
        Ok(loaded_voice)
    }

    /// Uses the cached file if there is one, without checking for updates.
    /// On Windows, hf-hub 1.0 copies a cached file again on every download
    /// call (it can't make symlinks), so avoid the call when possible.
    async fn download_file(repo: &ModelRepo, name: &str) -> Result<PathBuf> {
        let cached = repo.download_file().filename(name).local_files_only(true).send().await;
        if let Ok(path) = cached {
            return Ok(path);
        }
        repo.download_file()
            .filename(name)
            .send()
            .await
            .map_err(|source| Error::Download { file: name.to_string(), source })
    }

    fn client() -> Result<HFClient> {
        let mut builder = HFClient::builder();
        if let Some(cache_dir) = hf_cache_dir() {
            builder = builder.cache_dir(cache_dir);
        }
        builder.build().map_err(|source| Error::Download {
            file: format!("{}/{}", Self::MODEL_REPO_ORG, Self::MODEL_REPO_NAME),
            source,
        })
    }

    /// The names of the voices in the model repo, sorted, e.g. `af_heart`.
    ///
    /// If Hugging Face can't be reached, lists the voices downloaded so far
    /// instead (all of them, if the whole repo was ever downloaded), as the
    /// Python backend does.
    pub async fn list_voices() -> Result<Vec<String>> {
        let client = Self::client()?;
        let repo = client.model(Self::MODEL_REPO_ORG, Self::MODEL_REPO_NAME);
        let mut voices = match Self::list_repo_voices(&repo).await {
            Ok(voices) => voices,
            Err(source) => {
                let repo_folder =
                    format!("models--{}--{}", Self::MODEL_REPO_ORG, Self::MODEL_REPO_NAME);
                let cached = cached_voices(&client.cache_dir().join(repo_folder))
                    .ok_or(Error::ListVoices(source))?;
                log::warn!("cannot reach the voice repository; listing the cached voices");
                cached
            }
        };
        voices.sort();
        Ok(voices)
    }

    async fn list_repo_voices(repo: &ModelRepo) -> hf_hub::HFResult<Vec<String>> {
        let entries: Vec<RepoTreeEntry> =
            repo.list_tree().path_in_repo("voices").send()?.try_collect().await?;
        Ok(entries
            .into_iter()
            .filter_map(|entry| match entry {
                RepoTreeEntry::File { path, .. } => voice_file_name(path.strip_prefix("voices/")?),
                _ => None,
            })
            .collect())
    }

    /// Loads the model and voice, downloading them on first use.
    ///
    /// - `model` names a file in the model repo's `onnx/` folder, e.g.
    ///   `model` (full precision) or `model_quantized` (smaller and faster).
    /// - `language` is a code like `en-us`, `en-gb`, `es`, `fr-fr`, `hi`,
    ///   `it`, `pt-br` or `ja`. If `None`, it is taken from the voice name,
    ///   which isn't possible for `Voice::Array`.
    /// - `speed` is 1.0 for normal speed.
    pub async fn new(
        voice: Voice,
        model: &str,
        language: Option<&str>,
        speed: f32,
    ) -> Result<Self> {
        let language = match (language, &voice) {
            (Some(language), _) => language,
            (None, Voice::Name(name)) => voice_language(name)?,
            (None, Voice::Blend(parts)) => match parts.first() {
                Some(part) => voice_language(&part.name)?,
                None => return Err(Error::EmptyBlend),
            },
            (None, Voice::Array(_)) => return Err(Error::MissingLanguage),
        };

        let repo = Self::client()?.model(
            Self::MODEL_REPO_ORG,
            Self::MODEL_REPO_NAME,
        );
        let model_filename = format!("onnx/{model}.onnx");
        let (model_path, voice, g2p) = tokio::try_join!(
            Self::download_file(&repo, &model_filename),
            Self::get_voice(&repo, voice),
            G2p::new(language),
        )?;

        let session = Session::builder()?
            .commit_from_file(model_path)?;

        Ok(Self {
            voice,
            g2p,
            speed,
            session,
        })
    }

    /// Synthesizes `text` chunk by chunk. Chunks are produced lazily, so the
    /// caller can play the first one while later ones are still being made.
    ///
    /// `{name}` tags in the text aren't spoken; each is reported in
    /// [`Chunk::tags`] with the time of the word that follows it.
    ///
    /// The iterator borrows the synthesizer but not `text` (`use<'s>`).
    pub fn synthesize<'s>(
        &'s mut self,
        text: &str,
    ) -> Result<impl Iterator<Item = Result<Chunk>> + use<'s>> {
        let (clean, tags) = strip_tags(text);
        let words = self.g2p.split_words(&clean);
        let tags = tag_word_indices(&words, tags);
        let pw = words
            .iter()
            .map(|w| self.g2p.phonemize(&w.surface))
            .collect::<Result<Vec<_>>>()?;
        let word_heteronyms = self.g2p.heteronyms(&words);
        let n_words = words.len();
        let chunks = pack_words(&words, &pw, Self::CHUNK_BUDGET);

        Ok(chunks.into_iter().map(move |(lo, hi)| {
            // A tag goes to the chunk containing its word; a tag after the
            // last word goes to the last chunk.
            let chunk_tags = tags
                .iter()
                .filter(|(ord, _)| (lo..hi).contains(ord) || (*ord == n_words && hi == n_words))
                .map(|(ord, name)| (ord - lo, name.clone()))
                .collect();
            self.synth_chunk(
                &clean,
                &words[lo..hi],
                &pw[lo..hi],
                &word_heteronyms[lo..hi],
                chunk_tags,
            )
        }))
    }

    fn synth_chunk(
        &mut self,
        text: &str,
        words: &[Word],
        pw: &[String],
        word_heteronyms: &[Option<Heteronym>],
        tags: Vec<(usize, String)>,
    ) -> Result<Chunk> {
        let base = words[0].range.start;
        let chunk_text = &text[base..words[words.len() - 1].range.end];
        let ps = self.g2p.phonemize(chunk_text)?;

        // Heteronyms can only be fixed where we know which phonemes are whose.
        let (ps, spans) = match align_words(pw, &ps) {
            Some(spans) => {
                let (ps, spans) = english::fix_heteronyms(&ps, &spans, pw, word_heteronyms);
                (ps, Some(spans))
            }
            None => {
                log::warn!(
                    "subtitle_align_fallback: no per-word timing for {:?} ({} words)",
                    chunk_text.chars().take(80).collect::<String>(),
                    words.len(),
                );
                (ps, None)
            }
        };

        let ModelOutput { audio, starts } = self.run_model(&ps)?;
        let total_time = *starts.last().unwrap();

        let char_offset = |byte: usize| text[base..byte].chars().count();
        let word_timings: Option<Vec<WordTiming>> = spans.map(|spans| {
            words
                .iter()
                .zip(spans)
                .map(|(word, (a, b))| WordTiming {
                    text: word.surface.clone(),
                    char_start: char_offset(word.range.start),
                    char_end: char_offset(word.range.end),
                    start: starts[a],
                    end: starts[b],
                })
                .collect()
        });

        let n = words.len().max(1);
        let tag_timings = tags
            .into_iter()
            .map(|(ord, name)| {
                let t = match &word_timings {
                    Some(wt) => wt.get(ord).map_or(total_time, |w| w.start),
                    // Without word timings, spread the tags proportionally.
                    None => total_time * ord.min(n) as f64 / n as f64,
                };
                (name, t)
            })
            .collect();

        let phoneme_timings = ps
            .chars()
            .enumerate()
            .map(|(i, p)| (p, starts[i + 1]))
            .collect();

        Ok(Chunk {
            audio,
            text: chunk_text.to_string(),
            phonemes: phoneme_timings,
            words: word_timings,
            tags: tag_timings,
        })
    }

    fn run_model(&mut self, phonemes: &str) -> Result<ModelOutput> {
        let ids: Vec<i64> = phonemes
            .chars()
            .filter_map(|p| VOCAB.get(&p).copied())
            .collect();
        let style_index = ids.len().min(self.voice.len_of(Axis(0)) - 1);
        let style = self.voice.index_axis(Axis(0), style_index).to_owned();

        let mut input_ids = Vec::with_capacity(ids.len() + 2);
        input_ids.push(0);
        input_ids.extend(&ids);
        input_ids.push(0);
        let n = input_ids.len();

        let outputs = self.session.run(ort::inputs![
            "input_ids" => Tensor::from_array(([1, n], input_ids))?,
            "style" => Tensor::from_array(style)?,
            "speed" => Tensor::from_array(([1], vec![self.speed]))?,
        ])?;
        let (_, audio) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, durations) = outputs[1].try_extract_tensor::<f32>()?;

        // Scale the predicted durations (start padding, one per phoneme, end
        // padding) so they add up to the actual audio length.
        let total: f64 = durations.iter().map(|&d| d as f64).sum();
        let audio_secs = audio.len() as f64 / Self::SAMPLE_RATE as f64;
        let unit = if total > 0.0 { audio_secs / total } else { 0.0 };

        let mut starts = Vec::with_capacity(phonemes.chars().count() + 1);
        let mut time = durations[0] as f64 * unit;
        starts.push(time);
        let mut valid_ix = 1;
        for p in phonemes.chars() {
            // Phonemes missing from the vocabulary were not sent to the model,
            // so they take no time.
            if VOCAB.contains_key(&p) {
                if let Some(&d) = durations.get(valid_ix) {
                    time += d as f64 * unit;
                }
                valid_ix += 1;
            }
            starts.push(time);
        }

        Ok(ModelOutput {
            audio: audio.to_vec(),
            starts,
        })
    }
}

/// The voice name in a file name like `af_heart.bin`.
fn voice_file_name(file_name: &str) -> Option<String> {
    file_name.strip_suffix(".bin").map(str::to_string)
}

/// The voices in the cached snapshot of the main branch, from a repo folder
/// in the Hugging Face cache (`<repo>/refs/main` names the snapshot in
/// `<repo>/snapshots/`). `None` if there is no such snapshot.
fn cached_voices(repo_folder: &std::path::Path) -> Option<Vec<String>> {
    let commit = std::fs::read_to_string(repo_folder.join("refs").join("main")).ok()?;
    let voices_folder = repo_folder.join("snapshots").join(commit.trim()).join("voices");
    let voices = std::fs::read_dir(voices_folder)
        .ok()?
        .filter_map(|entry| voice_file_name(entry.ok()?.file_name().to_str()?))
        .collect();
    Some(voices)
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
    dirs::home_dir().map(|home| home.join(".cache").join("huggingface").join("hub"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Applications need to move the synthesizer into async tasks or share
    /// it (e.g. behind a mutex), so it must be `Send` and `Sync`.
    #[test]
    fn synthesizer_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<KokoroSynthesizer>();
        assert_send_sync::<Chunk>();
        assert_send_sync::<Error>();
    }

    /// These fail before anything is downloaded.
    #[tokio::test]
    async fn reports_bad_voices_and_languages() {
        let new = |voice, language| KokoroSynthesizer::new(voice, "model", language, 1.0);
        assert!(matches!(
            new(Voice::Blend(Vec::new()), None).await,
            Err(Error::EmptyBlend)
        ));
        assert!(matches!(
            new(Voice::Array(Array3::zeros((1, 1, 256))), None).await,
            Err(Error::MissingLanguage)
        ));
        assert!(matches!(
            new(Voice::Name("xx_nobody".into()), None).await,
            Err(Error::UnknownVoiceLanguage(name)) if name == "xx_nobody"
        ));
        assert!(matches!(
            G2p::new("zh").await,
            Err(Error::UnsupportedLanguage(language)) if language == "zh"
        ));
    }

    #[test]
    fn lists_cached_voices_of_the_main_snapshot() {
        let repo = std::env::temp_dir()
            .join(format!("kokoro-timestamped-test-{}", std::process::id()));
        assert_eq!(cached_voices(&repo), None);
        for (commit, file) in [("new", "af_heart.bin"), ("new", "README.md"), ("old", "af_old.bin")] {
            let voices = repo.join("snapshots").join(commit).join("voices");
            std::fs::create_dir_all(&voices).unwrap();
            std::fs::write(voices.join(file), b"").unwrap();
        }
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join("main"), "new\n").unwrap();
        let voices = cached_voices(&repo);
        std::fs::remove_dir_all(&repo).unwrap();
        assert_eq!(voices, Some(vec!["af_heart".to_string()]));
    }

    /// Needs Hugging Face (or a cached copy of the repo), so it only runs on
    /// request: `cargo test -- --ignored list_voices`.
    #[tokio::test]
    #[ignore]
    async fn list_voices_finds_voices_of_every_language() {
        let voices = KokoroSynthesizer::list_voices().await.unwrap();
        assert!(voices.is_sorted());
        for voice in ["af_heart", "bf_emma", "ef_dora", "ff_siwis", "hf_alpha", "if_sara", "jf_alpha", "pf_dora"] {
            assert!(voices.iter().any(|v| v == voice), "{voice} missing from {voices:?}");
        }
    }

    const FRAME: f64 = 0.01; // seconds

    /// Audio level in 10 ms frames (root mean square).
    fn envelope(audio: &[f32]) -> Vec<f64> {
        let frame = (FRAME * KokoroSynthesizer::SAMPLE_RATE as f64) as usize;
        audio
            .chunks(frame)
            .map(|c| (c.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / c.len() as f64).sqrt())
            .collect()
    }

    /// Checks phoneme, word and tag timings against each other and against
    /// the audio, for one voice per language. Downloads voices and runs the
    /// model, so it only runs on request: `cargo test -- --ignored timings`.
    #[tokio::test]
    #[ignore]
    async fn timings_are_sane_in_all_languages() {
        // Each text has three sentences, and tags at the start, before the
        // first word of the second sentence, and at the end.
        let cases = [
            ("af_heart", "{a} The weather is lovely today. {b} Look at that huge cat! It is sleeping on the keyboard, again. {c}"),
            ("bf_emma", "{a} The weather is lovely today. {b} Look at that huge cat! It is sleeping on the keyboard, again. {c}"),
            ("ef_dora", "{a} Hoy hace un día precioso. {b} ¡Mira ese gato enorme! Está durmiendo sobre el teclado, otra vez. {c}"),
            ("ff_siwis", "{a} Il fait très beau aujourd'hui. {b} Regarde ce gros chat ! Il dort sur le clavier, encore. {c}"),
            ("if_sara", "{a} Oggi è una bellissima giornata. {b} Guarda quel gatto enorme! Sta dormendo sulla tastiera, di nuovo. {c}"),
            ("pf_dora", "{a} Hoje está um dia lindo. {b} Olha aquele gato enorme! Ele está dormindo no teclado, de novo. {c}"),
            ("hf_alpha", "{a} आज मौसम बहुत अच्छा है। {b} उस बड़ी बिल्ली को देखो! वह फिर से कीबोर्ड पर सो रही है। {c}"),
            ("jf_alpha", "{a} 今日はとてもいい天気です。{b} あの大きな猫を見て！またキーボードの上で寝ています。{c}"),
        ];

        let mut failures = Vec::new();
        for (voice, text) in cases {
            let mut synthesizer = KokoroSynthesizer::new(Voice::Name(voice.into()), "model", None, 1.0)
                .await
                .unwrap();
            let chunks: Vec<Chunk> = synthesizer.synthesize(text).unwrap().map(Result::unwrap).collect();
            assert_eq!(chunks.len(), 1, "{voice}: expected one chunk");
            let chunk = &chunks[0];
            let mut fail = |msg: String| failures.push(format!("{voice}: {msg}"));

            // Phonemes: in order, and within the audio.
            let duration = chunk.audio.len() as f64 / KokoroSynthesizer::SAMPLE_RATE as f64;
            let ends: Vec<f64> = chunk.phonemes.iter().map(|&(_, t)| t).collect();
            let last = *ends.last().unwrap();
            if ends.windows(2).any(|w| w[1] < w[0]) {
                fail("phoneme times go backwards".into());
            }
            if ends[0] <= 0.0 || last > duration {
                fail(format!("phonemes end {:.3}..{last:.3}s, audio is {duration:.3}s", ends[0]));
            }

            // Words: aligned, in order, not overlapping.
            let Some(words) = &chunk.words else {
                fail("word alignment failed".into());
                continue;
            };
            if words.windows(2).any(|w| w[1].start < w[0].end - 1e-9 || w[0].start > w[0].end) {
                fail("word times overlap or go backwards".into());
            }

            // Tags: on the first word, the first word of the second
            // sentence, and the end.
            let sentence_starts: Vec<usize> = (1..words.len())
                .filter(|&i| words[i - 1].text.trim_end_matches(['"', '»', '”']).ends_with(['.', '!', '?', '。', '！', '？', '।']))
                .collect();
            let tag = |name: &str| chunk.tags.iter().find(|(n, _)| n == name).map(|&(_, t)| t);
            for (name, want) in [
                ("a", words[0].start),
                ("b", words[sentence_starts[0]].start),
                ("c", last),
            ] {
                if tag(name) != Some(want) {
                    fail(format!("tag {name} at {:?}, expected {want:.3}", tag(name)));
                }
            }

            // Against the audio: between two sentences there must be a
            // silence, and the next sentence must start where the audio does.
            let env = envelope(&chunk.audio);
            let frame_at = |t: f64| ((t / FRAME) as usize).min(env.len());
            let speech = env[frame_at(words[0].start)..frame_at(last)].iter().sum::<f64>()
                / (frame_at(last) - frame_at(words[0].start)) as f64;
            let silent = |f: usize| env[f] < 0.1 * speech;
            let mut onset_errors = Vec::new();
            let mut held = Vec::new();
            for &i in &sentence_starts {
                let (prev, next) = (&words[i - 1], &words[i]);
                // The pause is the longest silent stretch around the boundary
                // (consonants make short ones too).
                let mut longest = (0, 0); // frames, end exclusive
                let mut run_start = None;
                for f in frame_at(prev.start)..=frame_at(next.start + 0.15) {
                    match (f < env.len() && silent(f), run_start) {
                        (true, None) => run_start = Some(f),
                        (false, Some(s)) => {
                            if f - s > longest.1 - longest.0 {
                                longest = (s, f);
                            }
                            run_start = None;
                        }
                        _ => {}
                    }
                }
                let (gap_start, gap_end) = longest;
                if gap_end - gap_start < 5 {
                    fail(format!("no pause after {:?}", prev.text));
                    continue;
                }
                let onset = gap_end as f64 * FRAME;
                onset_errors.push(next.start - onset);
                // how long the previous word is held after its audio stopped
                held.push(prev.end - gap_start as f64 * FRAME);
            }
            if let Some(&worst) = onset_errors.iter().max_by(|a, b| a.abs().total_cmp(&b.abs())) {
                if worst.abs() > 0.1 {
                    fail(format!("sentence starts {worst:+.3}s from where the audio resumes"));
                }
            }

            println!(
                "{voice:9} audio {duration:.2}s, last phoneme ends {last:.2}s, {} words; \
                 next sentence starts vs audio onset {onset_errors:+.3?}s; \
                 sentence-final word held into the pause {held:.2?}s",
                words.len(),
            );
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
