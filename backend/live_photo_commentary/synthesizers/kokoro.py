import sys
import re
import logging
from pathlib import Path
import importlib
from importlib.metadata import distribution, PackageNotFoundError

from misaki.espeak import EspeakWrapper
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
import numpy as np
import onnxruntime

from ..synthesizer import Synthesizer
from ..text_segmentation import strip_tags, split_words, pack_words, align_words

logger = logging.getLogger(__name__)


class KokoroSynthesizer(Synthesizer):
    # from https://huggingface.co/hexgrad/Kokoro-82M/blob/4a34baf21f785e7cd08e41a290dffeb1fd0b66b7/config.json
    VOCAB = {';': 1, ':': 2, ',': 3, '.': 4, '!': 5, '?': 6, '—': 9, '…': 10, '"': 11, '(': 12, ')': 13, '“': 14, '”': 15, ' ': 16, '\u0303': 17, 'ʣ': 18, 'ʥ': 19, 'ʦ': 20, 'ʨ': 21, 'ᵝ': 22, '\uAB68': 23, 'A': 24, 'I': 25, 'O': 31, 'Q': 33, 'S': 35, 'T': 36, 'W': 39, 'Y': 41, 'ᵊ': 42, 'a': 43, 'b': 44, 'c': 45, 'd': 46, 'e': 47, 'f': 48, 'h': 50, 'i': 51, 'j': 52, 'k': 53, 'l': 54, 'm': 55, 'n': 56, 'o': 57, 'p': 58, 'q': 59, 'r': 60, 's': 61, 't': 62, 'u': 63, 'v': 64, 'w': 65, 'x': 66, 'y': 67, 'z': 68, 'ɑ': 69, 'ɐ': 70, 'ɒ': 71, 'æ': 72, 'β': 75, 'ɔ': 76, 'ɕ': 77, 'ç': 78, 'ɖ': 80, 'ð': 81, 'ʤ': 82, 'ə': 83, 'ɚ': 85, 'ɛ': 86, 'ɜ': 87, 'ɟ': 90, 'ɡ': 92, 'ɥ': 99, 'ɨ': 101, 'ɪ': 102, 'ʝ': 103, 'ɯ': 110, 'ɰ': 111, 'ŋ': 112, 'ɳ': 113, 'ɲ': 114, 'ɴ': 115, 'ø': 116, 'ɸ': 118, 'θ': 119, 'œ': 120, 'ɹ': 123, 'ɾ': 125, 'ɻ': 126, 'ʁ': 128, 'ɽ': 129, 'ʂ': 130, 'ʃ': 131, 'ʈ': 132, 'ʧ': 133, 'ʊ': 135, 'ʋ': 136, 'ʌ': 138, 'ɣ': 139, 'ɤ': 140, 'χ': 142, 'ʎ': 143, 'ʒ': 147, 'ʔ': 148, 'ˈ': 156, 'ˌ': 157, 'ː': 158, 'ʰ': 162, 'ʲ': 164, '↓': 169, '→': 171, '↗': 172, '↘': 173, 'ᵻ': 177}
    MODEL_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped"
    VOICE_LANGS = {
        "a": "en-us",
        "b": "en-gb",
        "j": "ja",
        "z": "zh",
        "e": "es",
        "f": "fr-fr",
        "h": "hi",
        "i": "it",
        "p": "pt-br",
    }
    _EXTRA_RE = re.compile(r"^(?P<dep>[^;\s]+)\s*;\s*extra\s*==\s*'(?P<extra>[^']+)'$")
    _MISAKI_LANGS = ['ja', 'zh']
    _PHONEME_CAPACITY = 512 - 2
    _CHUNK_BUDGET = 500  # pack words up to this many phoneme chars per chunk

    _g2p = None

    def __init__(self, voice="af_heart", model="model", lang=None, speed=1.0, **kwargs):
        super().__init__()
        self.__class__._init_misaki_capabilities()

        self.lang = lang
        self.speed = speed
        if isinstance(voice, np.ndarray):
            if lang is None:
                raise ValueError("With a binary voice, lang must be provided")
            self.voice = voice
        else:
            if isinstance(voice, str):
                voice_ratios = { voice: 1.0 }
            elif isinstance(voice, dict):
                voice_ratios = voice
            else:
                raise TypeError("voice must be a name, a ratio dictionary, or an ndarray")
            if not voice_ratios:
                raise TypeError("voice dict must not be empty")

            self.voice = None
            for voice_name, voice_ratio in voice_ratios.items():
                try:
                    voice_path = hf_hub_download(
                        repo_id=self.MODEL_REPO,
                        filename= f"voices/{voice_name}.bin",
                    )
                except Exception: # TODO make specific
                    raise ValueError(f"cannot find voice {voice_name} in {self.MODEL_REPO}")
                voice = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256) * voice_ratio
                if self.voice is None:
                    self.voice = voice
                else:
                    self.voice += voice

            if lang is None:
                self.lang = self.VOICE_LANGS[next(iter(voice_ratios))[0]]

        if self.lang.startswith('ja'):
            from unidic import download
            dicdir = Path(download.__file__).parent / 'dicdir'
            if not dicdir.is_dir():
                download.download_version()

        self.g2p = self.g2p_for(self.lang)

        model_path = hf_hub_download(
            repo_id=self.MODEL_REPO,
            filename=f"onnx/{model}.onnx",
        )
        providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)

    def _g2p_one(self, text: str) -> str:
        if not text:
            return ''
        phonemes, _ = self.g2p(text)
        return phonemes.strip()

    def __call__(self, text):
        clean, tag_ords = strip_tags(text)
        words = split_words(clean, self.lang)
        if not words:
            return
        pw_all = [self._g2p_one(w.surface) for w in words]
        n_words = len(words)
        for lo, hi in pack_words(words, pw_all, self._CHUNK_BUDGET):
            chunk_text = clean[words[lo].cstart:words[hi - 1].cend]
            chunk_tags = []
            for ordv, name in tag_ords:
                if lo <= ordv < hi:
                    chunk_tags.append((ordv - lo, name))
                elif ordv == n_words and hi == n_words:
                    chunk_tags.append((hi - lo, name))
            yield self._synth_chunk(chunk_text, words[lo:hi], pw_all[lo:hi], chunk_tags)

    def synthesize(self, text):
        clean, tag_ords = strip_tags(text)
        words = split_words(clean, self.lang)
        if not words:
            return np.zeros(1, dtype=np.float32), '', [], None, []
        pw_all = [self._g2p_one(w.surface) for w in words]
        tags = [(min(o, len(words)), name) for o, name in tag_ords]
        return self._synth_chunk(clean, words, pw_all, tags)

    def _synth_chunk(self, chunk_text, words, pw, tags):
        ps = self._g2p_one(chunk_text)
        audio, starts = self._run_model(ps)   # starts: len(ps)+1 absolute times
        total_time = starts[-1] if starts else 0.0

        spans = align_words(pw, ps)
        word_timings = None
        if spans is None:
            logger.warning(
                "subtitle_align_fallback: no per-word timing for %r (%d words)",
                chunk_text[:80], len(words),
            )
        if spans is not None:
            base = words[0].cstart
            word_timings = []
            for w, (a, b) in zip(words, spans):
                word_timings.append((
                    w.surface, w.cstart - base, w.cend - base,
                    round(starts[a], 4), round(starts[b], 4),
                ))

        tag_timings = []
        n = max(len(words), 1)
        for local_ordinal, name in tags:
            if word_timings is not None and 0 <= local_ordinal < len(word_timings):
                t = word_timings[local_ordinal][3]
            elif word_timings is not None:
                t = round(total_time, 4)
            else:
                t = round(total_time * min(local_ordinal, n) / n, 4)
            tag_timings.append((name, t))

        phoneme_timings = [(p, round(starts[i + 1], 4)) for i, p in enumerate(ps)]
        return audio, chunk_text, phoneme_timings, word_timings, tag_timings

    def _run_model(self, phonemes):
        """Run Kokoro and return (audio, starts), where starts has len(phonemes)+1
        absolute times: starts[0] is when phonemes[0] begins (i.e. the model's
        leading-silence offset) and starts[i+1] is when phonemes[i] ends."""
        input_ids = [self.VOCAB[p] for p in phonemes if p in self.VOCAB]
        style = self.voice[min(len(input_ids), self.voice.shape[0] - 1)]
        input_ids_for_model = [[0, *input_ids, 0]]
        audio, pred_durs = self.session.run(
            None,
            dict(
                input_ids=np.array(input_ids_for_model, dtype=np.int64),
                style=style,
                speed=np.array([self.speed], dtype=np.float32)),
        )
        audio = np.asarray(audio)[0].astype(np.float32)
        pred_durs = np.asarray(pred_durs)[0].astype(np.float64)  # BOS + per-phoneme + EOS
        # Scale the predicted durations so they sum to the true audio length.
        # This replaces a hand-tuned divisor and folds in the leading-silence
        # token pred_durs[0] as the offset before the first phoneme.
        total = float(pred_durs.sum())
        unit = (len(audio) / self.sample_rate()) / total if total else 0.0
        starts = [pred_durs[0] * unit]
        valid_ix = 1
        for phoneme in phonemes:
            dur = 0.0
            if phoneme in self.VOCAB:
                if valid_ix < len(pred_durs):
                    dur = pred_durs[valid_ix] * unit
                valid_ix += 1
            starts.append(starts[-1] + dur)
        return audio, starts

    def sample_rate(self):
        return 24000

    @classmethod
    def list_voices(cls):
        try:
            voices_path = Path('voices')
            files = list_repo_files(cls.MODEL_REPO)
            voices = [
                file_path.stem
                for file_path in map(Path, files)
                if file_path.parent == voices_path and file_path.suffix == '.bin'
            ]
        except Exception as x:
            local_path = Path(snapshot_download(cls.MODEL_REPO, local_files_only=True))
            voices = [
                file_path.stem
                for file_path in (local_path / 'voices').glob('*.bin')
            ]
            print(f"Warning: cannot reach voice repository: {x}", file=sys.stderr)

        return sorted(voices)

    @classmethod
    def _package_extras(cls, package, filter_fn):
        all_extras = set()
        incomplete_extras = set()
        deps = {}
        package_dist = distribution(package)
        for req in package_dist.requires or []:
            if match := cls._EXTRA_RE.match(req):
                extra = match.group('extra')
                if not filter_fn(extra):
                    continue
                all_extras.add(extra)
                dep = match.group('dep')
                if dep not in deps:
                    try:
                        distribution(dep)
                        deps[dep] = True
                    except PackageNotFoundError:
                        deps[dep] = False
                if not deps[dep]:
                    incomplete_extras.add(extra)
                    continue
        installed_extras = all_extras - incomplete_extras
        return all_extras, installed_extras

    @classmethod
    def list_langs(cls):
        cls._init_misaki_capabilities()
        return sorted(cls._g2p.keys())

    @classmethod
    def _init_misaki_capabilities(cls):
        if cls._g2p is not None:
            return

        misaki_extras, misaki_installed_extras = (
            cls._package_extras("misaki", lambda e: len(e) == 2)
        )

        langs = set(voice.language for voice in EspeakWrapper().available_voices())
        to_delete = set()
        cls._g2p = { lang: False for lang in [*langs, "en-us", "en-gb"] }
        for lang in cls._MISAKI_LANGS:
            if lang in misaki_installed_extras:
                cls._g2p[lang] = True
            elif lang in cls._g2p:
                to_delete.update(l for l in langs if l.startswith(lang))
        for lang in to_delete:
            del cls._g2p[lang]

    def g2p_for(self, lang):
        uses_extra = self._g2p.get(lang)
        if uses_extra is None:
            raise ValueError(f"Error: language {lang} is not supported by Espeak")
        if uses_extra:
            extra, *params = lang.split('-', 1)
            module_name = f"misaki.{extra}"
            # We are not using `en.G2P`, but just in case...
            if extra == 'en':
                cls_name = "G2P"
                british = params and params[0] != 'us'
                params = [None, False, british]
            else:
                cls_name = f"{extra.upper()}G2P"
        else:
            module_name = "misaki.espeak"
            cls_name = "EspeakG2P"
            params = [lang]
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        g2p = cls(*params)
        return g2p


if __name__ == '__main__':
    import ast
    import argparse
    import sounddevice as sd
    import time


    def play(audio, timings):
        sd.play(audio, 24000)
        start = time.perf_counter()
        for phoneme, timing in timings:
            print(phoneme, end='', flush=True)
            to_wait = timing + start - time.perf_counter()
            if to_wait > 0:
                time.sleep(to_wait)
        sd.wait()
        print()

    def synth_and_play(voice, title, text, lang=None):
        print(title)
        synthesizer = KokoroSynthesizer(voice=voice, lang=lang)
        for audio, fragment, phoneme_timings, word_timings, tag_timings in synthesizer(text):
            print(fragment)
            if word_timings:
                print("  ".join(w[0] for w in word_timings))
            if tag_timings:
                print(tag_timings)
            play(audio, phoneme_timings)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--voice",
        required=True,
        help="""Select voice. per https://huggingface.co/hexgrad/Kokoro-82M/blob\
                /main/VOICES.md""",
    )
    parser.add_argument(
        "-l",
        "--lang",
        help="""Select language, per https://github.com/espeak-ng/espeak-ng/blob\
                /master/docs/languages.md""",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="The text to speak",
    )
    parser.add_argument(
        "-V",
        "--voices",
        action="store_true",
        help="""List available voices""",
    )
    parser.add_argument(
        "-L",
        "--langs",
        action="store_true",
        help="""List available languages""",
    )

    args = parser.parse_args()

    if args.voices:
        print('\n'.join(KokoroSynthesizer.list_voices()))
        sys.exit()
    if args.langs:
        print('\n'.join(KokoroSynthesizer.list_langs()))
        sys.exit()

    voice = args.voice
    if args.text is None:
        print('Error: text required', file=sys.stderr)
        sys.exit(-1)

    title = voice
    if voice.startswith('{'):
        voice = ast.literal_eval(voice)

    synth_and_play(voice, title, args.text, args.lang)
