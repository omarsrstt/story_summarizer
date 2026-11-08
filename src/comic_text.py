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


# -- per-series glossaries --------------------------------------------------
#
# The glossary is the one piece of comic config that is *not* about the site: it
# names the series' own proper nouns, so one list cannot serve two comics. The
# tracked ``comic_config.json`` keeps a flat ``glossary`` for the series it was
# written for, and a ``glossaries`` map keyed by series name lets a second
# series carry its own without disturbing the first.
#
# The flat list stays the fallback rather than being migrated away, because a
# single-series setup is the common case and it must keep working untouched.


def resolve_glossary(config, series: str | None = None) -> list[str]:
    """The glossary terms for ``series``, from a parsed comic config.

    Lookup order, and why:

    1. ``glossaries[series]`` -- an explicit per-series list.
    2. ``glossaries["default"]`` -- shared terms (a publisher that prefixes every
       title, say), used only when the series has no list of its own.
    3. The flat ``glossary`` list -- the series ``comic_config.json`` was
       written for.

    A series with no entry therefore falls back to the flat list, which is what
    keeps this backwards compatible. Passing ``series=None`` skips straight to
    the flat list, so a caller that does not know the series is not silently
    given a different series' names.

    The series key is compared case-insensitively after the ``comic-`` prefix is
    stripped, because the reader takes ``--series TBATE`` while the transcript
    directory is ``comic-TBATE`` and the manifest field is ``TBATE``. Requiring
    the caller to know which spelling the config used would be a trap.
    """

    if not isinstance(config, dict):
        return []

    flat = _term_list(config.get("glossary"))
    if series is None:
        return flat

    keys = config.get("glossaries")
    if not isinstance(keys, dict):
        return flat

    # Both spellings of the series name: "TBATE" and "comic-TBATE". Folded so a
    # config written by hand in either case finds the entry.
    wanted: list[str] = []
    bare = series.strip()
    if bare.lower().startswith("comic-"):
        bare = bare[len("comic-"):]
    for spelling in (series.strip(), bare, "comic-" + bare):
        folded = spelling.casefold()
        if folded and folded not in wanted:
            wanted.append(folded)

    for key, value in keys.items():
        if str(key).casefold() in wanted:
            return _term_list(value)
    if "default" in keys:
        return _term_list(keys["default"])
    return flat


def _term_list(value) -> list[str]:
    """A glossary value as a list of non-empty strings.

    A config typo turns into an empty list rather than a ``TypeError`` from
    inside the segmenter: this is read on every transcript, and a malformed
    entry should cost one series its names, not abort a run that could still
    produce text.

    Entries are filtered on the *original* value, not on ``str(value)``: the
    latter turns a JSON ``null`` into the term ``"None"``, which would then be
    protected as if it were a character's name.
    """

    if not isinstance(value, list):
        return []
    return [
        str(term).strip()
        for term in value
        if isinstance(term, (str, int, float)) and str(term).strip()
    ]


def words_in(text: str) -> int:
    """Count real words, so a fused block is not counted as one."""

    total = 0
    for run in _RUN.findall(text):
        if _should_split(run):
            total += len(wordninja.split(run.lower()))
        elif not _CONTRACTION.match(run) or len(run) > 1:
            total += 1
    return total


# -- credit and watermark lines ---------------------------------------------
#
# A chapter's first and last pages carry the scan group's credits: a URL, a
# social handle, and a "Role: Name" block. They are the page's producer, not its
# story, and they were 24 of the 97 suspicious tokens in a 7-chapter corpus.
#
# The rule keys on *structure*, never on credit vocabulary, because the
# vocabulary is ordinary English. Measured on the real corpus, "to control MANA
# is" and "HAVING THE control of" are dialogue, and a rule matching "control"
# or "original" deletes them. Structure separates the two cases cleanly.

# A domain, a protocol, or a shortener. A URL is furniture wherever it appears.
# ``test`` is in the list because that is this project's reserved placeholder
# TLD (see comic_site.example.json): a rule that cannot recognise its own
# placeholder is awkward to test without naming a real site.
_URL = re.compile(
    r"(?:https?://|www\.|\b[a-z0-9-]+\.(?:com|net|org|ly|io|me|test)\b)",
    re.IGNORECASE,
)

# A social handle. Requires two characters after the @ so a stray mark is not
# read as a handle.
_HANDLE = re.compile(r"@[A-Za-z0-9_]{2,}")

# "Role: Name", the shape of a credit block. Both sides must be short and
# capitalised like a title, and the left side must be a job anyone would
# recognise -- this is what stops "He said: I will not go." being dropped.
#
# Spaces are optional inside a role because the recogniser drops them:
# "ComicArtist:Fuyuki23" is a single glued run, and "QualtyControl" is a
# misspelt "Quality Control". Both were measured in the real corpus.
#
# No publisher or site name appears here. An earlier draft matched "tapas",
# "isekai" and "tbate"; deleting all three changed *nothing* on the real corpus,
# because the URL rule and the trailing "entertainment" already covered them.
# A generic pipeline rule must not need this one series' brands.
_CREDIT_ROLE = re.compile(
    r"^\s*(?:\()?(?:"
    r"original(?:\s|$)|\s*novel|"
    r"comic\s*artist|"
    r"qual[a-z]*\s*control|"
    r"art\s*by|"
    r"story\s*by|"
    r"presented\s*by|"
    r"translat(?:or|ion)|"
    r"entertainment"
    r")",
    re.IGNORECASE,
)

# A line that ends as a bare role, which is how a credit block is sometimes
# split across two lines ("quality control by"), or that ends with the
# publisher's name ("KISAI ENTERTAINMENT", "(kisa i entertainment)"). "original"
# is deliberately not here: "NOT VERY / ORIGINAL." is dialogue, measured on page
# 50 of chapter 181.
_CREDIT_TAIL = re.compile(
    r"\b(?:quality\s*control(?:\s+by)?|entertainment)\s*\)?\s*\.?\s*$",
    re.IGNORECASE,
)

# Four words is the longest real hit ("Quality Control: Kisai Entertainment").
# The guard used to be six, which also caught ordinary sentences: "I do it for
# entertainment." is five words and ends the same way. Kept tight because the
# measured evidence is 2-4 words, never more.
_CREDIT_TAIL_WORDS = 4

# The credits page also writes "role/name": "lining assistant/MIKAGE",
# "coloring assistants/an sean de fu aka". Requiring the role word is what
# keeps this safe -- "Earthborm/vil doral" is a map label and must survive.
_CREDIT_SLASH = re.compile(
    r"^\s*[a-z ]{0,20}(?:assistant|editor|artist|colorist|letterer)s?\s*/",
    re.IGNORECASE,
)

# A role line sometimes stands alone with its value on the next line
# ("ART BY" / "TURTLEME"). The whole line must therefore be a role *phrase*,
# not merely contain a job word: "ART IS LOOKING" and "to even control" are
# dialogue, and a single-word match deleted "AT HIMSELF IN" and "the bodily
# functions" from the real corpus. Kept as whole phrases so it stays exact.
_ROLE_LINE = re.compile(
    r"^\s*(?:"
    r"art\s+by|story\s+by|script\s+by|"
    r"presented\s+by|"
    r"(?:original\s+)?novel(?:\s+and\s+script)?|"
    r"quality\s*control(?:\s+by)?|"
    r"comic\s*artist|"
    r"(?:main|storyboard|coloring|lining|render|background)?\s*"
    r"assistants?|"
    r"(?:assistant\s+)?editors?|"
    r"translat(?:or|ion)(?:\s+by)?"
    r")\s*[.:]?\s*$",
    re.IGNORECASE,
)


def _looks_like_job_line(text: str) -> bool:
    """Whether ``text`` is *only* a role line, e.g. ``ART BY`` or ``presented by``."""

    return bool(_ROLE_LINE.match(text.strip()))


def _credit_role_lead(stripped: str) -> bool:
    """Whether a line opens with a credit role *and* is shaped like a credit.

    The role regex is a prefix match, so on its own it also matches prose that
    merely begins with a role word: "Original sin was the topic", "Story by the
    fire, he said", "Art by the river was nothing special". None of those occur
    in the measured corpus, only because comic dialogue is mostly shouted and
    short -- which is luck, not a guarantee.

    A real credit line is a *label*, and only two shapes are labels: a colon
    introduces the role's value ("Quality Control: Kisai Entertainment"), or the
    role runs to the end of the line ("STORY BY") and is at least two words. A
    one-word line is excluded outright: "ORIGINAL" is the punchline of "NOT VERY
    / ORIGINAL." on page 50 of chapter 181, and the raw corpus contains no
    one-word role label that needs dropping.
    """

    match = _CREDIT_ROLE.match(stripped)
    if not match:
        return False
    if ":" in stripped[match.end():]:
        return True
    return match.end() >= len(stripped) and len(stripped.split()) >= 2


def is_credit_line(text: str, previous: str | None = None) -> bool:
    """Whether a line is scan-group furniture rather than story text.

    Four shapes are recognised, all measured against a hand-read corpus:

    * a URL or domain -- ``scan-site.test``, ``bit.ly/some-volume``
    * a social handle -- ``@turtleme93``
    * a credit block -- ``ComicArtist:Fuyuki23``, ``Quality Control: Kisai
      Entertainment``; the OCR drops the space, so the role arrives glued to
      its value and only the leading role word is reliable
    * ``role/name`` or ``role`` followed by a name on the next line -- ``lining
      assistant/MIKAGE``, ``ART BY`` / ``TURTLEME``

    Credit *vocabulary* is deliberately not a signal. "control", "original" and
    "comic" all occur in dialogue, so matching them would delete real text; the
    role must be at the start of the line and be a recognisable job title.

    ``previous`` is the preceding line of the same transcript, used only for the
    last shape. Passing it is optional so a single block can still be judged.
    """

    stripped = text.strip()
    if not stripped:
        return False
    # Two or three letters are a panel sound effect or detector noise, not a
    # credit block: "ACK!" must survive.
    if _URL.search(stripped) or _HANDLE.search(stripped):
        return True
    if _credit_role_lead(stripped):
        return True
    if _CREDIT_SLASH.match(stripped):
        return True
    if len(stripped.split()) <= _CREDIT_TAIL_WORDS and _CREDIT_TAIL.search(stripped):
        return True
    # A bare name is only a credit when a role line precedes it. Checked last
    # because it is the only shape that depends on context, and guarded twice:
    # the line must be short *and* contain no ordinary English word. Without the
    # second guard, "it is imperative" follows "assistant editors" and is
    # deleted -- measured in chapter 181, where it is the first line of dialogue
    # after the credits page.
    #
    # An "all capitals" escape was tried for publisher names made of ordinary
    # words ("TAPAS MEDIA") and removed: it changed nothing on the real corpus
    # (the pages were already scrubbed, so it could never be measured), while a
    # short shout after a role line is exactly what it would have deleted.
    if previous and _looks_like_job_line(previous):
        words = stripped.split()
        if len(words) <= 3 and not any(
            word.strip(".,:()&").lower().replace("'", "") in _DICTIONARY
            for word in words
        ):
            return True
    return False


def keep_credit_mask(lines) -> list[bool]:
    """Which lines survive :func:`scrub_credit_lines`, positionally.

    Exposed separately because a caller holding structured blocks needs to know
    *which* lines went, not just what the text became. Matching the surviving
    text back to blocks by string is wrong as soon as a line repeats, and
    credits contain repeats ("tapas" appears twice on one page).
    """

    mask: list[bool] = []
    previous: str | None = None
    for line in lines:
        if is_credit_line(line, previous=previous):
            previous = line
            mask.append(False)
            continue
        mask.append(True)
        previous = line
    return mask


def scrub_credit_lines(lines) -> list[str]:
    """Drop scan-group credit lines from a chapter's lines.

    Each line is judged on its own, with the previous line as the only context
    (a name follows ``ART BY``). A *block* rule was tried first -- once a credit
    is found, keep removing short lines until one is long -- and it was wrong:
    the real corpus lost ``THE MIRROR!``, ``RUMBLE``, ``VIRION!`` and ``CAN'T
    BLAME YOURSELF.`` because every one of them is two words. A credit block is
    not reliably contiguous at the text level, so the safer rule is the local
    one. Names split onto their own line are left behind as ``noise`` warnings,
    which is the honest cost of not deleting story text.
    """

    return [line for line, keep in zip(lines, keep_credit_mask(lines)) if keep]