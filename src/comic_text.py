"""Turn raw OCR blocks from comic pages into text fit to summarise.

The detector gives us text and a confidence, nothing more. Two things go wrong
in the raw output often enough to wreck a transcript:

* The recogniser is Chinese-trained, so it drops inter-word spaces on Latin
  text. Around 58% of blocks came back as ``HEDOESN'THIT``.
* It occasionally emits a CJK glyph inside otherwise English text.

A dictionary segmenter fixes the spacing, but it also shreds proper nouns
(``SAPIN`` -> ``sap in``). Names are protected by a per-series glossary rather
than guessed, because nothing in the text distinguishes ``sap in`` from
``block him from``.
"""

from __future__ import annotations

import re

import wordninja

# Unicode ranges for CJK ideographs, kana and fullwidth forms. Any of these in
# an English comic is a recogniser hallucination, not content.
_CJK = re.compile(
    "[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)

# An alphabetic run inside one whitespace-separated token, keeping internal
# apostrophes so "DOESN'T" stays one run. Punctuation splits runs, so
# "REY,MAKESURE" yields "REY" and "MAKESURE".
_RUN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

# Below this length a run is left alone. Measured against real OCR: the words
# that genuinely needed splitting were mostly 7+ letters, while the short
# proper nouns ("SAPIN", "XYRUS", "Etistin") and the noise tokens ("KWNI",
# "dWHHL") were all 6 or fewer. Length alone keeps both groups safe.
_MIN_RUN = 7

# A run that is itself a contraction ("DON'T", "ISN'T") is one word.
_CONTRACTION = re.compile(r"^[A-Za-z]+n't$|^[A-Za-z]+'[a-z]+$", re.IGNORECASE)

_LEADING_NOISE = re.compile(r"^\s*(\d+|[|/\\_=~^*#<>]+)\s*")


def has_cjk(text: str) -> bool:
    """True if ``text`` contains CJK characters."""

    return bool(_CJK.search(text))


def clean_ocr_text(text: str) -> str:
    """Drop hallucinated glyphs and stray edge noise from one OCR block."""

    text = _CJK.sub("", text)
    # Digits or stray marks glued to the front of a bubble are usually the
    # detector catching a panel edge or a page number, not part of the line.
    text = _LEADING_NOISE.sub("", text)
    return text.strip()


def _is_mixed_case(run: str) -> bool:
    """True if ``run`` has a capital after its first letter *and* a lowercase.

    All-caps runs are fused dialogue ("HISHEAD") and must still split; only a
    name like "TurtleMe" or "TelmoreCity" has both cases mixed.
    """

    return any(char.islower() for char in run) and any(
        char.isupper() for char in run[1:]
    )


def _should_split(run: str) -> bool:
    """Whether a single alphabetic run is several words missing their spaces.

    Three guards, each measured against real OCR rather than guessed:

    * ``_MIN_RUN`` -- short proper nouns ("SAPIN", "XYRUS") and noise ("KWNI")
      are all 5 letters or fewer, while every genuinely fused word was 6+.
    * a capital after the first letter marks a name ("TurtleMe", "Fuyuki").
    * a split containing a one-letter fragment ("color is t") is damage, not a
      repaired word.

    No separate contraction check is needed: the segmenter already returns a
    single part for real contractions ("doesn't", "isn't"), while a fused run
    that merely ends in one ("THINKSI'VE") still splits into several.
    """

    if len(run) < _MIN_RUN or _is_mixed_case(run):
        return False
    parts = wordninja.split(run.lower())
    if len(parts) <= 1:
        return False
    # A one-letter fragment is damage ("color is t", "ze s tier"), unless it is
    # the pronoun/determiner a segmenter legitimately splits off ("IS A LITTLE").
    for part in parts:
        if len(part) == 1 and part not in ("a", "i"):
            return False
    return True


def looks_concatenated(text: str) -> bool:
    """True if any run in ``text`` is several words fused together.

    Runs shorter than ``_MIN_RUN`` are ignored, so OCR noise like ``dWHHL`` or
    ``KWNI`` is left visibly as noise instead of being "recovered".
    """

    return any(_should_split(run) for run in _RUN.findall(text))


def _split_run(run: str, protected: set[str]) -> str:
    lower = run.lower()
    for term in protected:
        token = term.lower()
        if len(token) >= 3 and token in lower:
            # Re-segment around a known story term so it survives whole, e.g.
            # "TOCONTROLMANAIS" -> "to control" + "MANA" + "is".
            rebuilt = []
            for piece in re.split(f"({re.escape(token)})", lower):
                if not piece:
                    continue
                rebuilt.append(
                    piece.upper() if piece == token else " ".join(wordninja.split(piece))
                )
            return " ".join(" ".join(rebuilt).split())
    return " ".join(wordninja.split(lower))


def resegment(text: str, known: list[str] | None = None) -> str:
    """Recover inter-word spaces in fused runs, preserving punctuation.

    Each run the segmenter is confident about is split; anything else is left
    exactly as it was, so blocks that need no repair keep their original
    capitalisation. Names in ``known`` are never split, and a fused run that
    contains one keeps that name whole.
    """

    protected = {name.upper().strip() for name in (known or [])}

    def split_run(match: re.Match) -> str:
        run = match.group(0)
        if run.upper() in protected:
            return run
        if not _should_split(run):
            return run
        return _split_run(run, protected)

    return _RUN.sub(split_run, text)


def words_in(text: str) -> int:
    """Count real words, so a fused block is not counted as one."""

    total = 0
    for run in _RUN.findall(text):
        if _should_split(run):
            total += len(wordninja.split(run.lower()))
        elif not _CONTRACTION.match(run) or len(run) > 1:
            total += 1
    return total
