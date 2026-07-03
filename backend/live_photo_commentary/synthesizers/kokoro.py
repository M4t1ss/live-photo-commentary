import sys
import re
from pathlib import Path
import importlib
from importlib.metadata import distribution, PackageNotFoundError

from semantic_text_splitter import TextSplitter
from misaki.espeak import EspeakWrapper
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
import numpy as np
import onnxruntime

from ..synthesizer import Synthesizer


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
    _SPLITTER_CACHE_SIZE = 10000
    _EXTRACT_MARKS_RE = re.compile(r'\{([^}]*)\}')
    _WHITESPACE_PLUS_RE = re.compile(r' {2,}')
    _SUB_TAG_RE = re.compile(r'\{sub\}')

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

    def _chunk_splitter_callback(self, text: str) -> int:
        # chunk splitting is based on phoneme count vs TTS capacity
        # deny splitting within tags
        if text.count('{') != text.count('}'):
            return self._PHONEME_CAPACITY + 1
        # ignore tag contents
        markless, _ = self._extract_marks(text)
        phonemes, _ = self.g2p(markless)
        return len(phonemes)

    def __call__(self, text):
        from ..subtitle_splitter import insert_subtitle_tags
        from .. import config

        capacity = self._PHONEME_CAPACITY
        while True:
            splitter = TextSplitter.from_callback(self._chunk_splitter_callback, capacity=capacity)
            chunks = list(splitter.chunks(text))
            prepared = []
            max_excess = 0
            for chunk in chunks:
                tagged = insert_subtitle_tags(chunk, max_chars=config.get().subtitle_max_chars)
                subtitle_segs = self._subtitle_segments(tagged)
                markless, marks = self._extract_marks(tagged)
                braceless = self._EXTRACT_MARKS_RE.sub('', markless).strip()
                braceless = self._WHITESPACE_PLUS_RE.sub(' ', braceless)
                phonemes, _ = self.g2p(markless)
                excess = len(phonemes) - self._PHONEME_CAPACITY
                max_excess = max(max_excess, excess)
                prepared.append((phonemes, marks, subtitle_segs, braceless))
            if max_excess <= 0:
                break
            capacity -= max_excess

        for args in prepared:
            yield self._synthesize_phonemes(*args)

    @classmethod
    def _extract_marks(cls, text):
        marks = []
        def extract(match):
            marks.append(match.group(1))
            return "{}"
        new_text = cls._EXTRACT_MARKS_RE.sub(extract, text)
        return new_text, marks

    # XXX: unneeded?
    @classmethod
    def _restore_marks(cls, text, marks):
        def restore(_):
            return "{" + marks.pop(0) + "}"
        return cls._EXTRACT_MARKS_RE.sub(restore, text)

    @classmethod
    def _subtitle_segments(cls, text: str) -> list[str]:
        """Split text on {sub} marks and strip all other tags from each segment."""
        parts = cls._SUB_TAG_RE.split(text)
        result = [cls._EXTRACT_MARKS_RE.sub('', part).strip() for part in parts]
        return [s for s in result if s]

    def synthesize(self, text):
        from ..subtitle_splitter import insert_subtitle_tags
        from .. import config
        tagged = insert_subtitle_tags(text, max_chars=config.get().subtitle_max_chars)
        subtitle_segs = self._subtitle_segments(tagged)
        markless, marks = self._extract_marks(tagged)
        braceless = self._EXTRACT_MARKS_RE.sub('', markless).strip()
        braceless = self._WHITESPACE_PLUS_RE.sub(' ', braceless)
        phonemes, _ = self.g2p(markless)
        return self._synthesize_phonemes(phonemes, marks, subtitle_segs, braceless)

    def _synthesize_phonemes(self, phonemes, marks, subtitle_segments, braceless_text):
        input_ids = []
        restore = []
        for ix, phoneme in enumerate(phonemes):
            if phoneme in self.VOCAB:
                input_ids.append(self.VOCAB[phoneme])
            else:
                restore.append((ix, phoneme))
        voice_style_input = self.voice[len(input_ids)]
        input_ids_for_model = [[0, *input_ids, 0]]
        try:
            audio, pred_durs = self.session.run(
                None,
                dict(
                    input_ids=np.array(input_ids_for_model, dtype=np.int64),
                    style=voice_style_input,
                    speed=np.array([self.speed], dtype=np.float32)),
            )
        except Exception as x:
            # TODO: do something
            raise

        pred_durs = np.asarray(pred_durs)[0] / 40 # magic divisor for kokoro timestamps
        full_durs = np.zeros(len(phonemes), dtype=pred_durs.dtype)
        valid_ix = 1 # skip first and last
        for ix in range(len(phonemes)):
            if phonemes[ix] in self.VOCAB:
                full_durs[ix] = pred_durs[valid_ix]
                valid_ix += 1
        timings = full_durs.cumsum().tolist()

        audio = np.asarray(audio)[0].astype(np.float32)
        phoneme_timings = list(zip(phonemes, timings))
        mark_timings = list(zip(marks, [
            timing for phoneme, timing in phoneme_timings if phoneme == '{'
        ]))
        phoneme_timings = [
            (phoneme, timing) for phoneme, timing in phoneme_timings
            if phoneme not in '{}'
        ]
        return audio, braceless_text, phoneme_timings, mark_timings, subtitle_segments

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
        for audio, fragment, phoneme_timings, mark_timings in synthesizer(text):
            print(fragment)
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
