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

# Below this length a run is only split when the split is *cheap*: every part
# must be a common English word (see ``_SPLIT_COST_LIMIT``). 7+ letters still
# split on the dictionary alone. Measured against real OCR: long fused runs were
# mostly 7+ letters, while short proper nouns ("SAPIN", "XYRUS") and the noise
# tokens ("KWNI", "dWHHL") were all 6 or fewer.
_MIN_RUN = 7

# A short run is split only when no part is an unusual word. This is what lets
# short fused dialogue be repaired without a glossary:
#
#     HEHAS  -> he has   (max cost 5.83)      SAPIN  -> sap in   (11.74)
#     IWAS   -> i was   (6.57)                XYRUS  -> xy rus   (12.41)
#     TOFORM -> to form (7.98)                ETISTIN-> et is tin(10.54)
#     ABIT   -> a bit   (9.62)                ELENOIR-> e lenoir (13.77)
#
# Fused sequences top out at 9.62 and proper nouns resume at 10.54 on the real
# corpus, so 10.0 sits in a measured gap rather than being a tuned knob. Only
# applies below ``_MIN_RUN``; 7+ letter runs keep the older dictionary rule.
_SPLIT_COST_LIMIT = 10.0

# Contraction tails are dictionary entries ("re", "ve", "ll") but are never
# words on their own. Splitting "THEYRE" into "they re" invents a token that was
# not in the page, so any split producing one of these is refused.
_FRAGMENTS = frozenset({"re", "ve", "ll", "nt", "s", "t", "d", "m"})

# English suffixes are not separate words either. Without this, a story term
# whose stem is a real word gets bisected: "AUGMENTER" -> "augment er",
# "AUGMENTERS" -> "augment ers". Guarded only for 7+ letter runs, which is where
# the earlier dictionary rule would otherwise have split them.
_SUFFIXES = frozenset({"er", "ers", "ing", "ed", "est", "ly"})

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


# The segmenter's own word list (126k entries). Used as evidence of whether a
# run is text at all: "kwni" and "dwhhl" are in no dictionary, so keeping them
# only clutters the transcript.
_DICTIONARY = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost


def _is_known_word(run: str) -> bool:
    return run.lower() in _DICTIONARY


def _is_mixed_case(run: str) -> bool:
    """True if ``run`` has a capital after its first letter *and* a lowercase.

    All-caps runs are fused dialogue ("HISHEAD") and must still split; only a
    name like "TurtleMe" or "TelmoreCity" has both cases mixed.
    """

    return any(char.islower() for char in run) and any(
        char.isupper() for char in run[1:]
    )


def _split_cost(parts: list[str]) -> float:
    """Worst (rarest) word cost among ``parts``; infinite if one is unknown."""

    return max((_DICTIONARY.get(part, float("inf")) for part in parts), default=float("inf"))


def _should_split(run: str) -> bool:
    """Whether a single alphabetic run is several words missing their spaces.

    The decisive test is whether the split produces real words. Measured on a
    real chapter, that separates the cases cleanly where length alone did not:

    * every genuine fused run split into dictionary words ("BLOCKHIMFROM" ->
      block/him/from, 143 runs)
    * noise and proper nouns did not ("KWNI", "dWHHL" split into non-words;
      "ELENOIR", "SAPIN" are in no English dictionary)

    A capital after the first letter marks a name outright ("TurtleMe"), and a
    split containing a one-letter fragment ("color is t") is damage rather than
    a repaired word. Contractions need no special case: the segmenter returns a
    single part for "doesn't", while "THINKSI'VE" still splits.

    Three guards make this safe for *short* runs, which the dictionary alone
    cannot judge:

    * a run that is already a word is never two words ("MANA", "ALICE")
    * a split producing a contraction tail or a bare suffix is refused
      ("THEYRE" -> "they re", "AUGMENTER" -> "augment er")
    * below ``_MIN_RUN`` the split must also be *cheap* (all common words), which
      is what separates "HEHAS" -> "he has" from "SAPIN" -> "sap in"
    """

    low = run.lower()
    # An existing word is never a fused pair. "MANA" is a dictionary word but
    # the segmenter offers "man a"; the glossary handles the fused case where
    # MANA sits inside a longer run.
    if low in _DICTIONARY:
        return False
    if len(run) < 3 or _is_mixed_case(run):
        return False
    parts = wordninja.split(low)
    if len(parts) <= 1:
        return False
    if any(part in _FRAGMENTS for part in parts):
        return False
    if any(len(part) == 1 and part not in ("a", "i") for part in parts):
        return False
    # Every part must be a real word, otherwise this is a name or noise and
    # splitting it can only make it worse.
    if not all(_is_known_word(part) for part in parts):
        return False
    if len(run) >= _MIN_RUN:
        # Long runs keep the original rule, but never shave a suffix off a stem:
        # "AUGMENTER" is reducible to "augment" + "er" and must survive whole.
        if len(parts) == 2 and parts[1] in _SUFFIXES:
            return False
        return True
    return _split_cost(parts) <= _SPLIT_COST_LIMIT


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


def is_meaningful(text: str, known: list[str] | None = None) -> bool:
    """Whether a block contains any real text rather than detector noise.

    Comic pages carry specular highlights, page numbers and panel borders that
    the detector reads as glyphs. Keeping those only clutters a transcript the
    summariser has to read, so a block is kept only if it contains a dictionary
    word, a glossary term, or a run that splits into real words.
    """

    protected = {name.lower().strip() for name in (known or [])}
    for run in _RUN.findall(text):
        low = run.lower()
        if low in protected:
            return True
        # Single letters are in the dictionary ("m", "o"), so a lone one is
        # page furniture rather than a word.
        if len(low) >= 2 and _is_known_word(low):
            return True
        parts = wordninja.split(low)
        if (
            len(parts) > 1
            and all(len(part) > 1 or part in ("a", "i") for part in parts)
            and all(_is_known_word(part) for part in parts)
        ):
            return True
    return False


def words_in(text: str) -> int:
    """Count real words, so a fused block is not counted as one."""

    total = 0
    for run in _RUN.findall(text):
        if _should_split(run):
            total += len(wordninja.split(run.lower()))
        elif not _CONTRACTION.match(run) or len(run) > 1:
            total += 1
    return total
