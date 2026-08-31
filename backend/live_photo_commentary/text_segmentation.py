"""Turn VLM response text into timed, word-segmented subtitle data.

The pipeline used to smuggle emotion-tag positions and subtitle breaks through
Kokoro by leaving literal ``{...}`` in the text handed to the g2p (misaki/espeak
preserve them) and reading back the cumulative time at each ``{``. The braces
perturbed espeak's prosody and gave no per-word timing.

This module strips the tags up front, splits the clean text into words (a
whitespace split for most languages, a Sudachi-based chunker for Japanese), and
provides :func:`align_words` to map each word onto a span of the phoneme string
that Kokoro actually synthesises -- so the synthesiser can attach a start/end
time to every word without a single brace reaching the g2p.
"""

import re
from dataclasses import dataclass

_TAG_RE = re.compile(r"\{([^}]*)\}")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\S+")

_SENTENCE_END = tuple(".?!。！？")  # . ? ! 。 ！ ？
_CLOSERS = "\"'”』」)]}"           # closing quotes/brackets after the stop

# Phoneme diacritics that isolated g2p keeps but in-context g2p often drops;
# ignored when scoring the alignment so they don't dominate the edit distance.
_NORM_DROP = str.maketrans("", "", "ˈˌː")  # ˈ ˌ ː


@dataclass
class Word:
    surface: str
    cstart: int
    cend: int


def strip_tags(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Remove every ``{tag}`` and report where each one sat.

    Returns ``(clean, tags)`` where ``clean`` is the tag-free text with runs of
    whitespace collapsed, and ``tags`` is a list of ``(word_ordinal, name)``:
    ``word_ordinal`` is the number of clean words that precede the tag, i.e. the
    index of the word the tag should fire on (``len(clean.split())`` means "after
    the last word / at end of audio"). Tags are assumed to sit at word
    boundaries, as the VLM prompt instructs.
    """
    parts = _TAG_RE.split(text)  # [text, name, text, name, ..., text]
    clean_chunks: list[str] = []
    raw_tags: list[tuple[int, str]] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            prefix = _WS_RE.sub(" ", "".join(clean_chunks)).strip()
            raw_tags.append((len(prefix.split()), part))
        else:
            clean_chunks.append(part)
    clean = _WS_RE.sub(" ", "".join(clean_chunks)).strip()
    n_words = len(clean.split())
    tags = [(min(ordinal, n_words), name) for ordinal, name in raw_tags]
    return clean, tags


def split_words(clean: str, lang: str | None) -> list[Word]:
    """Split ``clean`` into words with character offsets into ``clean``."""
    if lang and lang.startswith("ja"):
        return [Word(clean[s:e], s, e) for s, e in chunk_japanese(clean)]
    return [Word(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(clean)]


def pack_words(words, pw, budget: int = 500) -> list[tuple[int, int]]:
    """Group word indices into ``(lo, hi)`` chunks whose combined isolated
    phoneme length stays under ``budget``, preferring to end a chunk right after
    a sentence-final word. ``pw[i]`` is word ``i`` phonemised in isolation.
    A single over-budget word becomes its own chunk.
    """
    result: list[tuple[int, int]] = []
    lo = 0
    size = 0
    brk = None  # exclusive index of the last sentence boundary in the current run
    for i in range(len(words)):
        w_size = len(pw[i]) + 1
        if size and size + w_size > budget and i > lo:
            hi = brk if (brk is not None and brk > lo) else i
            result.append((lo, hi))
            lo = hi
            size = sum(len(pw[k]) + 1 for k in range(lo, i + 1))
            brk = None
        else:
            size += w_size
        if words[i].surface.rstrip(_CLOSERS).endswith(_SENTENCE_END):
            brk = i + 1
    if lo < len(words):
        result.append((lo, len(words)))
    return result


# ── Japanese ─────────────────────────────────────────────────────────────────

_JA_LEFT_ATTACH = frozenset({"助詞", "助動詞", "接尾辞"})
_JA_NOUN = "名詞"
_JA_PREFIX = "接頭辞"
_JA_PUNCT = "補助記号"
_JA_SPACE = "空白"

_ja_tokenizer = None
_ja_split_mode = None


def _ja_setup():
    global _ja_tokenizer, _ja_split_mode
    if _ja_tokenizer is None:
        from sudachipy import Dictionary, SplitMode

        _ja_tokenizer = Dictionary().create()
        _ja_split_mode = SplitMode.C
    return _ja_tokenizer, _ja_split_mode


def chunk_japanese(text: str) -> list[tuple[int, int]]:
    """Segment Japanese text into word-like chunks, as ``(start, end)`` offsets.

    Sudachi mode C gives long units; on top of that we merge noun compounds,
    pull trailing particles / auxiliaries / suffixes onto the previous chunk,
    attach a leading prefix to the following chunk, and glue trailing
    punctuation onto the word before it (matching how "word." reads as one unit
    in Latin scripts). Leading/standalone punctuation stays on its own.
    """
    tokenizer, mode = _ja_setup()
    morphemes = list(tokenizer.tokenize(text, mode))
    pos = [m.part_of_speech()[0] for m in morphemes]

    chunks: list[tuple[int, int]] = []
    cur: list[int] | None = None       # [start, end] of the chunk being built
    held_prefix: list[int] | None = None  # a 接頭辞 waiting for the next chunk

    for i, m in enumerate(morphemes):
        p = pos[i]
        if p == _JA_SPACE:
            continue
        start, end = m.begin(), m.end()

        if p == _JA_PREFIX:
            if held_prefix is None:
                held_prefix = [start, end]
            else:
                held_prefix[1] = end
            continue

        if cur is None:
            cur = [held_prefix[0] if held_prefix else start, end]
            held_prefix = None
        else:
            cur[1] = end

        next_pos = None
        for j in range(i + 1, len(morphemes)):
            if pos[j] != _JA_SPACE:
                next_pos = pos[j]
                break

        if p == _JA_NOUN and next_pos == _JA_NOUN:
            continue
        if p != _JA_PUNCT and next_pos in _JA_LEFT_ATTACH:
            continue
        if p != _JA_PUNCT and next_pos == _JA_PUNCT:
            continue

        chunks.append((cur[0], cur[1]))
        cur = None

    if cur is not None:
        chunks.append((cur[0], cur[1]))
    elif held_prefix is not None:
        chunks.append((held_prefix[0], held_prefix[1]))
    return chunks


# ── Phoneme alignment ────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def _norm(s: str) -> str:
    return s.translate(_NORM_DROP).lower()


def align_words(pw: list[str], ps: str, window: int = 22):
    """Partition ``ps`` into ``len(pw)`` consecutive spans matching ``pw``.

    ``pw[k]`` is word ``k`` phonemised in isolation; ``ps`` is the whole chunk
    phonemised together (the string Kokoro synthesises). Walks a cursor along
    ``ps``, picking the local end that best matches each word's isolated
    phonemes. The search window is anchored between the cursor and a global
    proportional estimate, so cross-word coarticulation in ``ps`` (linking
    consonants, elisions) that knocks the cursor off can't strand it -- it is
    pulled back onto the diagonal within a word or two. Returns half-open
    ``(p_start, p_end)`` spans (trimmed of surrounding spaces), or ``None`` only
    if the total mismatch is so large the whole pairing must be wrong.
    """
    n = len(pw)
    L = len(ps)
    if n == 0 or L == 0:
        return None
    if n == 1:
        return [(0, L)]

    # `pw[k]` normalised once; `ps` is sliced RAW (indices must stay valid) and
    # normalised only inside the distance call.
    norm_pw = [_norm(token) for token in pw]
    total_norm = sum(len(t) for t in norm_pw) or 1
    ratio = L / total_norm  # ps runs longer than isolated words (stress/length marks, liaison)

    spans = []
    cursor = 0
    consumed = 0  # sum of norm_pw lengths for words already placed
    total_cost = 0.0
    for k in range(n):
        while cursor < L and ps[cursor] == " ":
            cursor += 1
        a = cursor
        target = norm_pw[k]
        remaining = n - 1 - k
        consumed += len(target)

        if k == n - 1:
            b = L
        else:
            g_end = consumed * ratio                       # global estimate of word end
            local_end = a + max(1, round(len(target) * ratio))
            lo = max(a + 1, round(min(local_end, g_end)) - window)
            hi = min(L - remaining, round(max(local_end, g_end)) + window)
            hi = min(hi, lo + 120)
            if hi <= lo:
                b = min(L - remaining, max(a + 1, local_end))
            else:
                b = lo
                best = None
                for cand in range(lo, hi + 1):
                    cost = _levenshtein(_norm(ps[a:cand]), target)
                    if cand < L and ps[cand] != " " and ps[cand - 1] != " ":
                        cost += 0.5  # prefer cutting on a word boundary
                    if best is None or cost < best:
                        best = cost
                        b = cand
                # A poor local match must not strand the cursor: snap toward the
                # global estimate instead of trusting the bad match.
                if best is not None and best > max(2.0, 0.6 * len(target)):
                    b = min(hi, max(lo, round(g_end)))
                total_cost += best or 0.0

        end = b
        while end > a and ps[end - 1] == " ":
            end -= 1
        spans.append((a, max(a, end)))
        cursor = b

    if total_cost > 0.6 * L:
        return None
    return spans
