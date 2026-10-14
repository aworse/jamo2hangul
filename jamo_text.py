"""Hangul <-> compatibility-jamo text transforms for the jamo->Hangul restoration task.

Input the model sees:  a flat stream of compatibility jamo (ㅎㅏㄴㄱㅡㄹ), spaces removed.
Target the model emits: the original text with spacing (한글 / 한 글).

The hard part is that the jamo stream is ambiguous: a consonant between two vowels can be the
final (받침) of the previous syllable or the initial of the next, and word spacing is not in the
input. A language model must resolve both. See the self-test (`python jamo_text.py`).
"""
from __future__ import annotations

# Compatibility jamo (U+3131..U+3163), ordered by the standard composition indices.
CHO = list("ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒ")  # placeholder; replaced below with compat forms
CHO = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JUNG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
        "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

SBASE, LCOUNT, VCOUNT, TCOUNT = 0xAC00, 19, 21, 28
NCOUNT = VCOUNT * TCOUNT          # 588
SCOUNT = LCOUNT * NCOUNT          # 11172

# All distinct compatibility jamo that can appear in a decomposed stream (for vocab / noise inventory).
JAMO_SET = sorted(set(CHO) | set(JUNG) | set(j for j in JONG if j))

_CHO_I = {c: i for i, c in enumerate(CHO)}
_JUNG_I = {c: i for i, c in enumerate(JUNG)}
_JONG_I = {c: i for i, c in enumerate(JONG)}  # includes "" -> 0


def is_syllable(ch: str) -> bool:
    return SBASE <= ord(ch) < SBASE + SCOUNT


def decompose_syllable(ch: str) -> str:
    """One Hangul syllable -> its 2 or 3 compatibility jamo (초성+중성[+종성])."""
    i = ord(ch) - SBASE
    l, v, t = i // NCOUNT, (i % NCOUNT) // TCOUNT, i % TCOUNT
    return CHO[l] + JUNG[v] + JONG[t]


def to_jamo(text: str, keep_space: bool = False) -> str:
    """Text -> flat compatibility-jamo stream. Hangul syllables are decomposed; every other character
    (digits, latin, punctuation, lone jamo) passes through unchanged. Spaces are dropped unless keep_space."""
    out = []
    for ch in text:
        if ch == " ":
            if keep_space:
                out.append(" ")
        elif is_syllable(ch):
            out.append(decompose_syllable(ch))
        else:
            out.append(ch)
    return "".join(out)


def compose_greedy(jamo: str) -> str:
    """Reference rule-based composer (NO language model): greedily attach a consonant as 받침 only when
    it cannot start a syllable (i.e. not followed by a vowel). Deterministic, used only as a baseline and
    to show why a model is needed — it cannot make the 한글/한 글 or 가나/간아 choice."""
    out, i, n = [], 0, len(jamo)
    while i < n:
        c = jamo[i]
        if c in _CHO_I and i + 1 < n and jamo[i + 1] in _JUNG_I:
            l, v = _CHO_I[c], _JUNG_I[jamo[i + 1]]
            i += 2
            t = 0
            # take a trailing consonant as 받침 only if it is not the initial of a following syllable
            if i < n and jamo[i] in _JONG_I and _JONG_I[jamo[i]] and not (i + 1 < n and jamo[i + 1] in _JUNG_I):
                t = _JONG_I[jamo[i]]
                i += 1
            out.append(chr(SBASE + (l * VCOUNT + v) * TCOUNT + t))
        else:
            out.append(c)
            i += 1
    return "".join(out)


def make_pair(text: str, keep_space_in_src: bool = False) -> tuple[str, str]:
    """(source jamo stream, target text). Target keeps original spacing; source drops it by default so the
    model must restore spacing from context."""
    return to_jamo(text, keep_space=keep_space_in_src), text


def corrupt(jamo: str, p_sub: float, p_del: float, p_ins: float, rng) -> str:
    """Inject jamo-level noise to make the model robust to an imperfect upstream (keyboard/acoustic) source.
    Operates only on jamo characters; pass-through characters are left alone. All rates default off."""
    if not (p_sub or p_del or p_ins):
        return jamo
    out = []
    for c in jamo:
        if c in JAMO_SET:
            r = rng.random()
            if r < p_del:
                continue
            if r < p_del + p_sub:
                c = rng.choice(JAMO_SET)
            out.append(c)
            if rng.random() < p_ins:
                out.append(rng.choice(JAMO_SET))
        else:
            out.append(c)
    return "".join(out)


if __name__ == "__main__":
    import random

    # 1) round-trip: decompose then NFC-compose a well-ordered sequence must return the original
    import unicodedata
    CHO_J = [chr(0x1100 + i) for i in range(19)]
    JUNG_J = [chr(0x1161 + i) for i in range(21)]
    JONG_J = [""] + [chr(0x11A8 + i) for i in range(27)]

    def conjoining(text):  # exact inverse check via Unicode NFC on conjoining jamo
        o = []
        for ch in text:
            if is_syllable(ch):
                i = ord(ch) - SBASE
                o.append(CHO_J[i // NCOUNT] + JUNG_J[(i % NCOUNT) // TCOUNT] + JONG_J[i % TCOUNT])
            else:
                o.append(ch)
        return unicodedata.normalize("NFC", "".join(o))

    samples = ["한글", "한 글", "가나", "간아", "안녕하세요 반갑습니다", "우리나라 대한민국",
               "오늘 날씨가 맑습니다", "학생이 책을 읽는다", "2026년 10월", "ChatGPT와 비교"]
    ok = all(conjoining(s) == s for s in samples)
    print("exact decompose->compose round-trip:", ok)

    print("\njamo stream (what the model sees) -> target:")
    for s in samples[:6]:
        src, tgt = make_pair(s)
        print(f"  {src:18s} -> {tgt}")

    print("\nambiguity (same jamo, different valid target) & greedy baseline:")
    for s in ["한글", "한 글", "가나", "간아"]:
        src = to_jamo(s)
        print(f"  {src:8s}  target={s!r:8}  greedy={compose_greedy(src)!r}")

    rng = random.Random(0)
    print("\nnoise injection (p_sub=p_del=p_ins=0.1):")
    for s in ["안녕하세요", "오늘 날씨가 맑습니다"]:
        print(f"  clean={to_jamo(s)}\n  noisy={corrupt(to_jamo(s), 0.1, 0.1, 0.1, rng)}")
    print("\njamo inventory size:", len(JAMO_SET))
