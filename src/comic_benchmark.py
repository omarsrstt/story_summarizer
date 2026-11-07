"""Novel-backed benchmark for the comic OCR pipeline.

The novel is a **test oracle, never an input**. Nothing here may write to a
transcript, and no page of this file may be imported by the pipeline: it exists
to *measure* the pipeline, and a measurement that changes what it measures is
worthless. The novel-free validator (`comic_check.py`) stays the shippable gate;
this is the local instrument that says whether its numbers mean anything.

Why a benchmark at all
----------------------
`comic_check` cannot see loss. Without a reference, a line the OCR dropped and a
line the adapter cut are identical findings. Against the novel we can at least
see which words the comic genuinely contains but the transcript mangled, and
that is the difference between "2% of tokens look odd" and "2% are real names
spelled wrong".

What is and is not measurable
-----------------------------
The comic is an **adaptation**: it drops, reorders and rewrites, and its chapters
are roughly half the novel's length. So *recall against the novel is not a
quality metric* -- a missing sentence is usually a legitimate cut, not a lost
sentence. Anything that reports coverage as a score would punish the adapter.
What survives that constraint is:

* **Name spelling** -- of the names the transcript produced that the novel also
  knows, how many are spelled the novel's way. The denominator is *recognised*
  names, not every capitalised token: dividing by all of them read 43% when the
  real figure was 96%, because detector junk ("KWNI") was counted as a
  misspelling.
* **Misreads** -- ``transcript -> novel`` one edit apart, with how many chapters
  from the aligned one the novel uses that name. A distance-1 match alone is not
  evidence: it paired the shout "RAAH" with "Riah", a character introduced 285
  chapters later.
* **Glossary gaps** -- names spelled correctly that the segmenter *would* split,
  so they need an entry. Only reported when ``resegment`` actually splits them;
  without that condition 10 of 11 candidates needed no entry at all.
* **Adaptation spellings** -- the comic and the novel differ on purpose. The
  glossary follows the comic, so these are reported and *not* treated as errors.
* **Explainability** -- of the tokens the novel-free validator calls
  "suspicious", how many map to a novel name. This is what turns a vague noise
  count into a named defect.
* **Unmatched chapters** -- alignment confidence, reported so a low-fidelity
  chapter is not blamed when the pairing is actually wrong.

Alignment is by **content, never by chapter number**. Measured offset drift on
this series: comics 1-3 -> novels 1-3 (offset 0), comic 147 -> novel 100
(offset 47), comic 181 -> novel 138 (offset 43). Bonus chapters have no
counterpart and must be reported *unmatched*, never as OCR failures.

The pairing is series-specific knowledge, so it lives in a **gitignored local
file** (`--pairing`), not in tracked code. Without one, every chapter is aligned
by content and the result is a measurement, not a claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import comic_check
from comic_check import parse_transcript, suspicious_tokens
from comic_text import _DICTIONARY, resegment

# Config paths resolve from the repo root, like every other script here.
REPO_ROOT = Path(__file__).resolve().parent.parent

# A word for comparison: letters and internal apostrophes. The novel uses curly
# quotes (U+2019), so apostrophes are normalised before matching -- otherwise
# "don't" in the novel never equals "don't" in a transcript.
_WORD = re.compile(r"[A-Za-z][A-Za-z']{2,}")

# Ordinary English carries no identity. Deliberately short: rare words are the
# signal, so removing the most common ones is enough to expose them.
_STOPWORDS = frozenset("""
the and that this with from they have been were will would could should there
their what when where which while about after before into over under than then
them these those your he's she's it's don't didn't doesn't isn't aren't wasn't
weren't haven't hasn't won't can't couldn't wouldn't shouldn't i'm i've i'll
i'd you've you'll he'd she'd we've we're they've they're that's there's here's
""".split())

# A chapter is only "matched" when the best candidate beats the runner-up by this
# much cosine. Measured: real pairs score 0.052-0.284 margin, the two bonus
# chapters score 0.004 and 0.003. Ten times apart, so the floor sits between.
MATCH_MARGIN = 0.02

# Name spelling is only reported above this many recognised names. Comic 1 has
# one, so its raw figure for 0% says nothing about the pipeline.
MIN_NAMES_FOR_FIDELITY = 10

# A near-miss is a name at most this many edits away. Two is deliberately tight:
# measured, distance-2 also matches "MBLE" -> "Myre" and "HUHPI" -> "Hahp", which
# are unrelated words. Only distance <= 1 is trusted as an *explanation*; the
# wider band is reported separately as a guess.
EXPLAIN_DISTANCE = 1
LOOSE_DISTANCE = 2

# A novel form only counts as a name for *flagging purposes* when it is Title
# Case and used at least this many times. Measured, the guard removes "Agh"
# (1 use) and "NOOO" (4 uses, all-caps), which the unguarded matcher reported as
# "misread names"; real names clear it easily (Elenoir 268, Sapin 132, Dawsid 19).
# Occurrences, not chapters, so a small test fixture behaves like the real novel.
MIN_NAME_OCCURRENCES = 2


def normalise_apostrophes(text: str) -> str:
    """Fold the curly quotes the novel uses onto ASCII.

    Without this the novel's "don't" (U+2019) and a transcript's "don't" are
    different strings, and every contraction counts as a mismatch.
    """

    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def words_in(text: str) -> list[str]:
    """Lowercased words, apostrophes folded, for comparison only."""

    return [word.lower() for word in _WORD.findall(normalise_apostrophes(text))]


def edit_distance(a: str, b: str, limit: int = LOOSE_DISTANCE) -> int:
    """Levenshtein distance, giving up once it exceeds ``limit``.

    The early exit matters: this runs against ~1000 novel names per token, so a
    full table for every pair would be slow for no gain. Anything beyond the
    limit is reported as ``limit + 1`` rather than the exact figure, because the
    caller only ever asks "is it close enough?".
    """

    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for row, char_a in enumerate(a, 1):
        current = [row]
        for column, char_b in enumerate(b, 1):
            current.append(min(
                previous[column] + 1,          # deletion
                current[column - 1] + 1,       # insertion
                previous[column - 1] + (char_a != char_b),  # substitution
            ))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


@dataclass
class NovelChapter:
    """A novel chapter reduced to what the benchmark needs."""

    number: int
    counts: Counter
    vector: dict[str, float] = field(default_factory=dict)


@dataclass
class ChapterResult:
    """One comic chapter measured against its best novel candidate."""

    comic_number: int
    best: int
    score: float
    second: int
    second_score: float
    matched: bool
    expected: int | None = None
    name_tokens: int = 0
    name_exact: int = 0
    name_misread: int = 0
    name_loose: int = 0
    name_unknown: int = 0
    suspicious: int = 0
    explained: int = 0
    guessed: int = 0
    unexplained: list[str] = field(default_factory=list)
    glossary_gaps: list[str] = field(default_factory=list)
    misread: dict[str, tuple[str, int | None]] = field(default_factory=dict)
    spelling_differs: dict[str, str] = field(default_factory=dict)
    wrong_pair: bool = False

    @property
    def margin(self) -> float:
        return self.score - self.second_score

    @property
    def recognised_names(self) -> int:
        """Name-shaped tokens that are a novel name, exact or one edit away.

        Garbage ("KWNI") and adaptation-only names both land outside this set,
        and neither can honestly be called a misspelling.
        """

        return self.name_exact + self.name_misread

    @property
    def spelling_accuracy(self) -> float | None:
        """Share of *recognised* names spelled the novel's way.

        Deliberately not ``exact / name_tokens``. That denominator includes
        detector junk, so chapter 2 read 53% when its real names were nearly all
        correct -- the figure described garbage, not spelling, and would have
        sent someone hunting a spelling bug that was not there.
        """

        if self.recognised_names < MIN_NAMES_FOR_FIDELITY:
            return None
        return self.name_exact / self.recognised_names

    @property
    def explained_rate(self) -> float | None:
        if not self.suspicious:
            return None
        return self.explained / self.suspicious


class NovelIndex:
    """The novel, indexed for content alignment and name lookup."""

    def __init__(self, directory: Path) -> None:
        self.chapters: dict[int, NovelChapter] = {}
        self.proper_nouns: Counter = Counter()
        self._spelling: dict[str, dict] = {}

        for path in sorted(directory.glob("chapter_*.txt")):
            number = _chapter_number(path)
            if number is None:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            tokens = _WORD.findall(normalise_apostrophes(text))
            counts = Counter(token.lower() for token in tokens)
            self.chapters[number] = NovelChapter(number=number, counts=counts)
            for token in tokens:
                # A name is a token the dictionary does not know, capitalised
                # somewhere. Lowercased variants are folded by the counter.
                if token.lower() in _DICTIONARY:
                    continue
                if not (token[0].isupper() or token.isupper()):
                    continue
                self.proper_nouns[token] += 1
                record = self._spelling.setdefault(
                    token.lower(), {"title": 0, "total": 0, "chapters": []}
                )
                record["total"] += 1
                # Title Case is what prose uses for a name; all-caps is the
                # comic's convention for a shout. Measured, this is the clean
                # separator between a name and a sound effect.
                if not token.isupper():
                    record["title"] += 1
                # Which chapters use it. A name whose only use is hundreds of
                # chapters from the aligned one is weak evidence: "Riah" first
                # appears in chapter 287 but was used to "explain" a shout in
                # comic 2, which aligns to chapter 2.
                if number not in record["chapters"]:
                    record["chapters"].append(number)

        if not self.chapters:
            raise SystemExit(f"no chapter_*.txt in {directory}")

        self._build_vectors()

        # Lowercased name -> the spelling the novel uses most. A transcript
        # matching the lowercase form is "exact" regardless of case, because
        # OCR capitalises erratically.
        self.name_forms: dict[str, str] = {}
        form_counts: dict[str, Counter] = {}
        for spelling, count in self.proper_nouns.items():
            form_counts.setdefault(spelling.lower(), Counter())[spelling] = count
        for lowered, forms in form_counts.items():
            self.name_forms[lowered] = forms.most_common(1)[0][0]

        # Names bucketed by first letter, so a near-miss search is ~40 candidates
        # instead of ~1000.
        self.names_by_initial: dict[str, list[str]] = {}
        for lowered in self.name_forms:
            self.names_by_initial.setdefault(lowered[:1], []).append(lowered)

    def is_reliable_name(self, lowered: str) -> bool:
        """Whether the novel's use of this form is a name rather than a noise.

        Without this guard the "misread" list was fiction. Measured, it reported:

            AGB -> Agh      novel uses "Agh" once, in one chapter
            NOOI -> NOOO    "NOOO" is 4/4 all-caps

        Neither is a name: both are sound effects standing beside OCR rubble
        (``REAARLAA,P AGB AP``). Pointing someone at those is worse than saying
        nothing, because "fix this name" is actionable and wrong.

        A name must be Title Case and used more than once. Both clauses are
        load-bearing: "Agh" is Title Case but appears once, and "NOOO" recurs but
        is never Title Case. Real names clear both easily -- Elenoir 268 uses,
        Sapin 132, Dawsid 19. Counting occurrences rather than chapters keeps the
        rule independent of corpus size, so a two-chapter fixture behaves the
        same way as the 516-chapter novel.
        """

        data = self._spelling.get(lowered)
        if data is None:
            return False
        if not data.get("title"):
            return False
        return data["total"] >= MIN_NAME_OCCURRENCES

    def _build_vectors(self) -> None:
        """Weight every chapter's words by how rare they are across the novel.

        Inverse document frequency is what makes this work on an adaptation: a
        name shared with the right chapter ("ELENOIR") outweighs a common verb
        shared with all 516, so alignment keys on the parts that identify a
        chapter rather than the parts every chapter has.
        """

        document_frequency: Counter = Counter()
        for chapter in self.chapters.values():
            document_frequency.update(chapter.counts.keys())

        total = len(self.chapters)
        # Smoothed inverse document frequency: log((1 + N) / (1 + df)) + 1.
        # The +1 floor matters. Raw log(N / (1 + df)) is zero for any word
        # appearing in half the chapters, so a two-chapter novel gives every word
        # weight zero and every cosine 0.0 -- measured, and it made alignment
        # impossible on any small corpus.
        self.idf = {
            word: math.log((1 + total) / (1 + frequency)) + 1.0
            for word, frequency in document_frequency.items()
        }

        for chapter in self.chapters.values():
            chapter.vector = {
                word: (1 + math.log(count)) * self.idf[word]
                for word, count in chapter.counts.items()
                if word in self.idf
            }

    def vector_for(self, text: str) -> dict[str, float]:
        counts = Counter(words_in(text))
        return {
            word: (1 + math.log(count)) * self.idf[word]
            for word, count in counts.items()
            if word in self.idf
        }

    @staticmethod
    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        shared = set(a) & set(b)
        if not shared:
            return 0.0
        dot = sum(a[word] * b[word] for word in shared)
        norm_a = math.sqrt(sum(value * value for value in a.values()))
        norm_b = math.sqrt(sum(value * value for value in b.values()))
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    def candidates(self, text: str) -> list[tuple[float, int]]:
        """Every novel chapter scored against ``text``, best first."""

        vector = self.vector_for(text)
        scored = [
            (self.cosine(vector, chapter.vector), number)
            for number, chapter in self.chapters.items()
        ]
        scored.sort(reverse=True)
        return scored

    def is_name(self, lowered: str) -> bool:
        """Whether the novel uses this exact form as a name.

        No reliability guard here on purpose: if the transcript spells a name the
        way the novel does, that is a correct transcription whether or not the
        novel uses it often.
        """

        return lowered in self.name_forms

    def nearest_name(
        self, token: str, limit: int = LOOSE_DISTANCE
    ) -> tuple[str, int] | None:
        """The closest *reliable* novel name to ``token``, or None.

        Bucketed by first letter: a misread almost always keeps the first
        character ("Earthborm"/"Earthborn"), and comparing every token against a
        thousand candidates is wasted work.

        Candidates must pass :meth:`is_reliable_name`. Without that guard the
        distance match happily pairs a sound effect with a one-off token
        ("AGB" -> "Agh"), and a named non-defect is worse than silence.
        """

        lowered = token.lower()
        best: tuple[str, int] | None = None
        for candidate in self.names_by_initial.get(lowered[:1], ()):
            if not self.is_reliable_name(candidate):
                continue
            distance = edit_distance(lowered, candidate, limit)
            if distance <= limit and (best is None or distance < best[1]):
                best = (candidate, distance)
        return best

    def chapters_using(self, lowered: str) -> list[int]:
        """Novel chapters that use this form, ascending."""

        record = self._spelling.get(lowered)
        return list(record["chapters"]) if record else []

    def nearest_use(self, lowered: str, chapter: int) -> int | None:
        """How many chapters from ``chapter`` this form is next used, or None.

        Attached to every finding because it is the difference between evidence
        and a coincidence. Measured on this corpus:

            Dawsid    aligned 100, nearest use 100   (0 away, strong)
            Earthborn aligned   2, nearest use 162   (160 away, still real)
            Riah      aligned   2, nearest use 287   (285 away, coincidence)

        A threshold here would be invented -- Earthborn is genuine at 160 while
        Riah is not at 285, and nothing cleanly separates them -- so the figure
        is reported and the reader judges.
        """

        uses = self.chapters_using(lowered)
        if not uses:
            return None
        return min(abs(use - chapter) for use in uses)


def _chapter_number(path: Path) -> int | None:
    """The integer in ``chapter_00100.txt``, or None if the name is not that shape."""

    digits = path.stem.split("_")
    if len(digits) != 2 or not digits[1].isdigit():
        return None
    return int(digits[1])


def load_pairing(path: Path | None) -> dict[int, int]:
    """Comic ordinal -> novel chapter, from a local hand-checked file.

    This is series knowledge, not pipeline knowledge: on this comic the offset
    drifts (0, then 47, then 43), so it cannot be derived by arithmetic. Keeping
    it in a gitignored file means the tool ships generic and the pairing stays
    where it belongs.
    """

    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    # ``pairs`` keeps room for a ``_comment`` key explaining the file. A bare
    # mapping is still accepted so an older hand-made file keeps working.
    pairs = data.get("pairs", data)
    return {
        int(comic): int(novel)
        for comic, novel in pairs.items()
        if not comic.startswith("_")
    }


def measure(
    index: NovelIndex,
    comic_number: int,
    text: str,
    expected: int | None = None,
    glossary: list[str] | None = None,
) -> ChapterResult:
    """Score one transcript against the novel.

    ``text`` is the transcript file's content, header and page markers included;
    the word tokeniser ignores both, and parsing it is what lets the novel-free
    validator's own notion of "suspicious" be reused rather than reimplemented.

    ``glossary`` must be the *same* list ``comic_check`` is given. Passing an
    empty one made the benchmark call every known name suspicious and then
    "explain" it with the novel, which double-counted the same term: its
    suspicious count no longer matched the validator's, so the two tools could
    not be compared.
    """

    scored = index.candidates(text)
    best_score, best = scored[0]
    if len(scored) > 1:
        second_score, second = scored[1]
    else:
        # A one-chapter novel has no runner-up. Treat the margin as maximal
        # rather than crashing, but say so: with nothing to compare against,
        # "confident" is not a claim this can make.
        second_score, second = 0.0, best
    matched = (best_score - second_score) >= MATCH_MARGIN

    result = ChapterResult(
        comic_number=comic_number,
        best=best,
        score=best_score,
        second=second,
        second_score=second_score,
        matched=matched,
        expected=expected,
    )
    if expected is not None and expected != best:
        result.wrong_pair = True

    parsed = parse_transcript(text)
    tokens = _WORD.findall(normalise_apostrophes(text))

    # Name-shaped means "could plausibly be a name": a token the dictionary does
    # not know that is capitalised somewhere. That catches real names and also
    # detector junk ("KWNI"), which is why the two are separated below instead of
    # being divided into one misleading percentage.
    name_tokens = [
        token
        for token in tokens
        if token.lower() not in _DICTIONARY
        and (token[0].isupper() or token.isupper())
    ]
    result.name_tokens = len(name_tokens)

    # The glossary is the comic's own spelling authority, so it is consulted
    # before the novel on a token it covers.
    protected = {term.lower() for term in (glossary or [])}

    for token in name_tokens:
        lowered = token.lower()
        if index.is_name(lowered):
            result.name_exact += 1
        elif lowered in protected:
            # The glossary spells it this way, so the transcript is correct.
            # "ASHBERN" is how the comic spells it -- raw OCR agrees at 0.96 --
            # so calling it a misread of the novel's "Ashber" would be the novel
            # overruling the comic, the inversion the oracle rule forbids.
            result.name_exact += 1
            other = index.nearest_name(token, EXPLAIN_DISTANCE)
            if other is not None and other[1] == 1:
                result.spelling_differs[token.upper()] = index.name_forms[other[0]]
        else:
            near = index.nearest_name(token, EXPLAIN_DISTANCE)
            if near is not None:
                result.name_misread += 1
                # Keep the pair *and its distance from the aligned chapter*.
                # "Earthborm -> Earthborn" tells someone what to fix; the
                # distance tells them whether to believe it ("RAAH -> Riah" is
                # 285 chapters away and is a shout, not a name).
                spelling = index.name_forms[near[0]]
                result.misread[token] = (
                    spelling,
                    index.nearest_use(near[0], best),
                )
            elif index.nearest_name(token, LOOSE_DISTANCE) is not None:
                result.name_loose += 1
            else:
                result.name_unknown += 1

    # A name the novel spells the transcript's way but the glossary omits is only
    # a *gap* if the segmenter would otherwise split it. The glossary's stated
    # job is to keep names whole, so a name `resegment` already leaves alone
    # ("Dawsid", "Eksire") needs no entry: adding it would change nothing, and
    # reporting it would be noise of the benchmark's own making. Measured on this
    # corpus, that filter drops 9 of 11 candidates and keeps "Goodsky" -> "good
    # sky" and "Dicathen" -> "dic a then", which are the ones that matter.
    seen_gaps: set[str] = set()
    for token in name_tokens:
        lowered = token.lower()
        if lowered in protected or lowered not in index.name_forms:
            continue
        spelling = index.name_forms[lowered]
        if resegment(spelling) == spelling:
            continue
        if lowered not in seen_gaps:
            seen_gaps.add(lowered)
            result.glossary_gaps.append(spelling)

    suspicious = suspicious_tokens(parsed, glossary or [])
    result.suspicious = len(suspicious)
    for token in suspicious:
        if index.is_name(token.lower()):
            result.explained += 1
            continue
        if index.nearest_name(token, EXPLAIN_DISTANCE) is not None:
            result.explained += 1
            continue
        if index.nearest_name(token, LOOSE_DISTANCE) is not None:
            # Counted apart from "explained" on purpose. Distance 2 also matches
            # "MBLE" -> "Myre", so a loose hit is a suggestion, not a finding;
            # folding it into the explained rate would inflate the score.
            result.guessed += 1
            continue
        result.unexplained.append(token)

    return result


def classify_unexplained(
    tokens: list[str], glossary: list[str]
) -> dict[str, list[str]]:
    """Group unknown tokens by the pipeline defect they most resemble.

    A flat "unexplained" list is not actionable: it puts a fused glossary term
    next to detector noise, and the two need opposite responses. The buckets use
    only what is measurable here.

    * ``segmented`` -- contains a glossary term glued to other letters
      (``OfElenoir``, ``XyrusCity``). The shipped splitter leaves these whole
      because their split cost exceeds ``_SPLIT_COST_LIMIT``, so this is the
      known fused-run class, not mystery text.
    * ``noise`` -- nothing recognises it. Real detector junk.

    Credits are *not* here: recognising the bare-name shape needs the preceding
    line, so they are detected per line in ``measure`` and reported separately.
    """

    protected = [term.upper() for term in glossary]
    buckets: dict[str, list[str]] = {"segmented": [], "noise": []}

    for token in tokens:
        upper = token.upper()
        # A glossary term inside a longer run is fusion, not a misspelling. The
        # term must be strictly shorter than the token, or a bare name would be
        # reported as its own segmentation defect.
        if any(term and term in upper and len(term) < len(upper) for term in protected):
            buckets["segmented"].append(token)
        else:
            buckets["noise"].append(token)
    return buckets


def report(
    results: list[ChapterResult],
    glossary: list[str],
) -> None:
    print(f"{'comic':>6} {'best':>5} {'score':>6} {'margin':>7} "
          f"{'names':>16} {'suspicious':>16}  note")

    for result in results:
        if not result.matched:
            note = "unmatched (no counterpart)"
            print(f"{result.comic_number:>6} {result.best:>5} {result.score:>6.3f} "
                  f"{result.margin:>7.4f} {'-':>16} {'-':>16}  {note}")
            continue

        # Exact/recognised, not exact/all. The denominator is the names the
        # benchmark could identify at all, so garbage does not masquerade as a
        # spelling failure.
        accuracy = result.spelling_accuracy
        names_text = (
            f"{result.name_exact}/{result.recognised_names}"
            if accuracy is not None
            else f"too few ({result.recognised_names})"
        )
        explained = (
            f"{result.explained}/{result.suspicious}"
            if result.suspicious
            else "none"
        )
        notes = []
        if result.wrong_pair:
            notes.append(f"PAIRING WRONG (expected {result.expected})")
        if accuracy is not None and accuracy < 0.9:
            notes.append(f"spelling {accuracy:.0%}")
        if result.name_misread:
            notes.append(f"{result.name_misread} misread")
        if result.guessed:
            notes.append(f"+{result.guessed} loose")
        print(f"{result.comic_number:>6} {result.best:>5} {result.score:>6.3f} "
              f"{result.margin:>7.4f} {names_text:>16} {explained:>16}  "
              f"{', '.join(notes)}")

    matched = [r for r in results if r.matched]
    if not matched:
        print("\nno chapter was confidently aligned; nothing to aggregate")
        return

    # Aggregate over confident matches only, so an unmatched bonus chapter does
    # not drag the pipeline's score down for a pairing reason.
    exact = sum(r.name_exact for r in matched)
    recognised = sum(r.recognised_names for r in matched)
    total_suspicious = sum(r.suspicious for r in matched)
    explained_total = sum(r.explained for r in matched)
    guessed = sum(r.guessed for r in matched)
    unexplained = [token for r in matched for token in r.unexplained]
    gaps = [name for r in matched for name in r.glossary_gaps]
    conflicts: dict[str, str] = {}
    for result in matched:
        conflicts.update(result.spelling_differs)
    misread: dict[str, tuple[str, int | None]] = {}
    for result in matched:
        misread.update(result.misread)

    print(f"\n{len(matched)} aligned chapter(s), {len(results) - len(matched)} "
          f"unmatched")
    if recognised >= MIN_NAMES_FOR_FIDELITY:
        print(f"  name spelling      {exact}/{recognised} "
              f"({100 * exact / recognised:.1f}% of recognised names exact)")
    else:
        print(f"  name spelling      too few recognised names ({recognised})")
    if total_suspicious:
        print(f"  suspicious explained {explained_total}/{total_suspicious} "
              f"({100 * explained_total / total_suspicious:.1f}%), "
              f"{guessed} loose guess(es)")
    print(f"  glossary terms     {len(glossary)}")

    # The two findings with a named fix are printed first and separately, because
    # they are the ones someone can act on today.
    if misread:
        print(f"\n=== misread names ({len(misread)}) ===")
        print("  transcript -> novel (1 edit); 'near' is how many chapters from")
        print("  the aligned one the novel uses it. Small is strong evidence:")
        for wrong, (right, distance) in sorted(misread.items()):
            where = "?" if distance is None else str(distance)
            print(f"  {wrong!r} -> {right!r}  near={where}")

    if gaps:
        unique = sorted(dict.fromkeys(gaps))
        print(f"\n=== glossary gaps ({len(unique)}) ===")
        print("  spelled correctly, but the segmenter would split them:")
        print(f"  {', '.join(unique)}")

    if conflicts:
        print(f"\n=== adaptation spellings ({len(conflicts)}) ===")
        print("  the comic and the novel spell these differently. The glossary")
        print("  follows the comic, so this is NOT an error -- do not 'fix' it:")
        for term, novel_form in sorted(conflicts.items()):
            print(f"  comic {term!r} vs novel {novel_form!r}")

    buckets = classify_unexplained(unexplained, glossary)
    if any(buckets.values()):
        print("\n=== unexplained tokens, by likely cause ===")
        for label, hint in (
            ("segmented", "fused run containing a known term"),
            ("noise", "no recognisable source"),
        ):
            group = buckets[label]
            if not group:
                continue
            counts = Counter(group)
            listed = ", ".join(
                f"{token!r}x{n}" if n > 1 else repr(token)
                for token, n in counts.most_common(12)
            )
            print(f"  {label:<10} {len(group):>3}  ({hint})")
            print(f"             {listed}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure comic transcripts against the novel they adapt.",
        epilog=(
            "The novel is a test oracle only: this reads the novel and the "
            "transcripts, and writes nothing."
        ),
    )
    parser.add_argument("-n", "--name", required=True,
                        help="Comic slug under novels/, e.g. comic-TBATE")
    parser.add_argument("--novel", required=True,
                        help="Novel directory the comic adapts")
    parser.add_argument("--pairing", default=None,
                        help="Local JSON of comic ordinal -> novel chapter")
    parser.add_argument("--json", action="store_true",
                        help="Emit one JSON document instead of a table")
    args = parser.parse_args(argv)

    comic_dir = REPO_ROOT / "novels" / args.name
    novel_dir = REPO_ROOT / args.novel
    if not comic_dir.is_dir():
        raise SystemExit(f"no transcripts at {comic_dir}")
    if not novel_dir.is_dir():
        raise SystemExit(f"no novel at {novel_dir}")

    pairing = load_pairing(REPO_ROOT / args.pairing if args.pairing else None)
    glossary = load_glossary(REPO_ROOT / "src" / "comic_config.json")
    index = NovelIndex(novel_dir)

    transcripts = sorted(comic_dir.glob("chapter_*.txt"))
    if not transcripts:
        raise SystemExit(f"no chapter_*.txt in {comic_dir}")

    results: list[ChapterResult] = []
    for path in transcripts:
        number = _chapter_number(path)
        if number is None:
            continue
        text = path.read_text(encoding="utf-8")
        results.append(
            measure(
                index, number, text,
                expected=pairing.get(number),
                glossary=glossary,
            )
        )

    if args.json:
        print(json.dumps({
            "name": args.name,
            "novel": args.novel,
            "chapters": [
                {
                    "comic": r.comic_number,
                    "best": r.best,
                    "score": round(r.score, 4),
                    "margin": round(r.margin, 4),
                    "matched": r.matched,
                    "expected": r.expected,
                    "pairing_wrong": r.wrong_pair,
                    "name_tokens": r.name_tokens,
                    "name_exact": r.name_exact,
                    "name_misread": r.name_misread,
                    "name_loose": r.name_loose,
                    "name_unknown": r.name_unknown,
                    "spelling_accuracy": (
                        None if r.spelling_accuracy is None
                        else round(r.spelling_accuracy, 4)
                    ),
                    "suspicious": r.suspicious,
                    "explained": r.explained,
                    "guessed": r.guessed,
                    "unexplained": r.unexplained,
                    "glossary_gaps": r.glossary_gaps,
                    "spelling_differs": r.spelling_differs,
                    "misread": r.misread,
                }
                for r in results
            ],
        }, indent=2))
    else:
        report(results, glossary)

    # A wrong pairing is a benchmark defect, not a pipeline failure, so it is
    # surfaced loudly but does not silently pass.
    return 1 if any(r.wrong_pair for r in results) else 0


def load_glossary(config: Path) -> list[str]:
    """The same glossary ``comic_check`` uses, so the two agree.

    Reusing ``comic_check.load_glossary`` rather than reading the key here keeps
    one definition of a term; the benchmark and the validator must classify the
    same word the same way or their counts cannot be compared.
    """

    if not config.is_file():
        return []
    return comic_check.load_glossary(config)


if __name__ == "__main__":
    sys.exit(main())
