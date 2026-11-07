"""Validate comic transcripts without needing the novel.

The novel is a test oracle, never an input: most comics have no associated
novel, so a check that needs one cannot tell you whether the pipeline is
healthy. This reads the transcripts already on disk and reports what is wrong
with them, in two severities:

``fail``
    Unambiguous. A CJK glyph is not story text in an English comic, a
    non-printable byte is not prose, and a run the shipped segmentation rule
    would split is either stale or a regression. A page marker pointing past the
    images on disk, and a chapter short of the page count its manifest recorded,
    are the same class -- the second is the real "lists 81 pages, has 80" defect
    in ``Chapter 144``.

``warn``
    A real smell that shares its shape with legitimate text. ``man a`` is what
    MANA became, but "a man a day" is ordinary English; ``KWNI`` is detector
    noise, but a rare name is indistinguishable from it. Warning on these and
    failing on them are different products, and failing would make the gate
    something to ignore.

Invention is treated as worse than loss, but only the *unambiguous* invention
classes fail. Loss is measured rather than gated: without a reference, a missing
line and a cut line produce identical findings, which is why the ``suspicious
rate`` is reported as a trend rather than a pass/fail.

Every check here was validated by disabling it and confirming a test fails, and
each guard that looked redundant was measured rather than assumed: the
``MANA``-only split is what the cost gate is calibrated on, and ``Chapter 144.0``
is skipped because it is a fingerprint-confirmed duplicate, not a hole.

**The ``fused`` check gates staleness, not correctness.** It compares a
transcript against the rule *as it is now*, so it fires when the rule changed and
the transcripts were not re-run. Re-run them and the text agrees with whatever
rule produced it -- a regression that made the rule worse is then invisible to
this check, and shows up only as a rise in the suspicious rate. Neither one
replaces ``tests/test_comic_text.py``, which pins the rule itself. Measured by
regressing ``_MIN_RUN``: the fused finding went silent after re-OCR, while 120
short splittable pairs were left glued.

    uv run python src/comic_check.py -n comic-TBATE
    uv run python src/comic_check.py -n comic-TBATE --comics comics/TBATE
    uv run python src/comic_check.py -n comic-TBATE --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import wordninja

from comic_text import (
    _DICTIONARY,
    _SPLIT_COST_LIMIT,
    _split_cost,
    has_cjk,
    keep_credit_mask,
    resegment,
)

# ``# TBATE chapter 147`` / ``# comic-TBATE chapter 1``. The number is the
# ordinal on disk, not the site's label.
_HEADER = re.compile(r"^#\s*\S*\s*chapter\s+(\d+)", re.IGNORECASE)
_PAGE_MARKER = re.compile(r"^---\s*page\s*(\d+)\s*---$", re.IGNORECASE)
_RUN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

# Control bytes other than tab. A transcript is UTF-8 prose; anything else is
# a decode fault that reached disk.
_NONPRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# A line repeated this often inside one chapter is an OCR loop, not a refrain.
_REPEAT_THRESHOLD = 5
_REPEAT_MIN_TOKENS = 1

# Credit lines are decided by ``comic_text.is_credit_line``, the same rule the
# reader uses to drop them. This module used to carry its own marker list --
# "quality control", "turtleme", "<site>" -- which drifted from the reader: it
# flagged lines the reader keeps and missed ones it drops, so the warning never
# described what the pipeline actually did. One rule, one place. The marker list
# also named a third-party site, which is never written into tracked code.


@dataclass
class Line:
    """One transcript line with the page it came from."""

    page: int | None
    text: str


@dataclass
class Transcript:
    """A parsed transcript: its chapter number, page markers and lines."""

    chapter: int
    pages: list[int] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class Finding:
    """One problem found in a transcript."""

    kind: str
    severity: str  # "fail" or "warn"
    detail: str
    example: str = ""
    count: int = 1


def parse_transcript(text: str) -> Transcript:
    """Parse ``text`` into a chapter number, its page markers and its lines."""

    chapter = 0
    pages: list[int] = []
    lines: list[Line] = []
    current: int | None = None

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            match = _HEADER.match(stripped)
            if match:
                chapter = int(match.group(1))
            continue
        marker = _PAGE_MARKER.match(stripped)
        if marker:
            current = int(marker.group(1))
            pages.append(current)
            continue
        lines.append(Line(page=current, text=stripped))

    return Transcript(chapter=chapter, pages=pages, lines=lines)


def load_glossary(path) -> list[str]:
    """Read the ``glossary`` list from a comic config, or ``[]`` if absent."""

    path = Path(path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    terms = data.get("glossary", []) if isinstance(data, dict) else []
    return [str(term) for term in terms]


def _tokens(text: str) -> list[str]:
    return _RUN.findall(text)


def _is_known(word: str, protected: set[str]) -> bool:
    """Whether ``word`` is ordinary text rather than detector noise.

    The apostrophe is tried *both* ways on purpose. The dictionary carries
    contractions spelled with one ("don't", "they're"), so stripping it first
    turned every contraction into a non-word -- "theyre" is in no dictionary --
    and a page of ordinary dialogue was reported as noise. It also carries some
    words glued ("dont"), so the stripped form is kept as a second chance.
    """

    low = word.lower()
    if low in _DICTIONARY or low in protected:
        return True
    without = low.replace("'", "")
    return without in _DICTIONARY or without in protected


def suspicious_tokens(transcript: Transcript, glossary) -> list[str]:
    """The tokens counted as suspicious, in order.

    Exposed because the *rate* alone cannot be compared across a change that
    removes words. Scrubbing credits lowered chapter 183 from 6 suspicious
    tokens to 4, yet its rate rose 3.75% -> 4.55%: the credits were ordinary
    English, so removing them shrank the denominator more than the numerator.
    Any report of progress has to quote the count as well as the share.
    """

    protected = {term.lower() for term in glossary}
    return [
        token
        for token in _tokens(transcript.text)
        if len(token) > 1 and not _is_known(token, protected)
    ]


def suspicious_rate(transcript: Transcript, glossary) -> float:
    """Share of tokens that are neither dictionary words nor glossary terms.

    This is the novel-free quality number: it needs no reference text, so it
    works on a comic whose source novel does not exist. It is a ratio, so it is
    only comparable between transcripts of similar length -- quote
    :func:`suspicious_tokens` alongside it when judging a change that also
    changes how many words survive.
    """

    tokens = _tokens(transcript.text)
    if not tokens:
        return 0.0
    return len(suspicious_tokens(transcript, glossary)) / len(tokens)


def _fused_run_findings(transcript: Transcript, glossary) -> list[Finding]:
    """Runs the shipped pipeline would split but that are still glued together.

    This is a staleness gate, so the test is literally "would the current code,
    with the current config, change this text?" -- ``resegment`` is called with
    the glossary for that reason. Calling it without the glossary reported
    ASHBERN, ETISTIN and DICATHEN as stale on chapter 2, which had just been
    re-OCR'd: they were split by the check, not by the pipeline.

    Glue alone is not enough to call it a fault, though. A 7+ letter run splits
    under the older long-run rule even when it is a name the glossary does not
    know, and the validator cannot tell a name from noise. So a token only
    counts when the split is *cheap* -- every part a common word, which is the
    same evidence the pipeline uses before it will split a short run.
    """

    stuck = []
    for token in _tokens(transcript.text):
        if len(token) <= 2:
            continue
        if resegment(token, glossary) == token:
            continue
        parts = wordninja.split(token.lower())
        if len(parts) < 2:
            continue
        # ``_split_cost`` is the pipeline's own measure, and it returns infinity
        # for a part that is not a word -- so this one gate rejects both a name
        # and a noise token, with no separate "is every part known" test.
        if _split_cost(parts) > _SPLIT_COST_LIMIT:
            continue
        stuck.append(token)

    if not stuck:
        return []
    counts = Counter(stuck)
    example = counts.most_common(1)[0][0]
    return [
        Finding(
            kind="fused",
            severity="fail",
            detail=(
                f"{len(stuck)} token(s) are plainly fused but the shipped rule "
                f"would split them: the transcript is stale or the pipeline "
                f"regressed"
            ),
            example=example,
            count=len(stuck),
        )
    ]


def _term_split_findings(transcript: Transcript, glossary) -> list[Finding]:
    """A glossary term that appears as two ordinary words.

    ``MANA`` -> ``man a`` is what the OCR emitted in the real corpus. It is a
    warning rather than a failure because the same two words occur legitimately
    in English ("a man a day").
    """

    findings: list[Finding] = []
    words = [w.lower() for w in _tokens(transcript.text)]

    for term in glossary:
        low = term.lower().strip()
        if " " in low:
            continue
        parts = wordninja.split(low)
        if len(parts) < 2:
            continue
        for index in range(len(words) - len(parts) + 1):
            if words[index : index + len(parts)] == parts:
                findings.append(
                    Finding(
                        kind="term_split",
                        severity="warn",
                        detail=(
                            f"glossary term {term!r} appears split as "
                            f"{' '.join(parts)!r}"
                        ),
                        example=" ".join(parts),
                    )
                )
                break
    return findings


def _noise_findings(transcript: Transcript, glossary) -> list[Finding]:
    protected = {term.lower() for term in glossary}
    bad = [
        token
        for token in _tokens(transcript.text)
        if len(token) > 1 and not _is_known(token, protected)
    ]
    if not bad:
        return []
    counts = Counter(token.lower() for token in bad)
    return [
        Finding(
            kind="noise",
            severity="warn",
            detail=f"{len(bad)} token(s) are neither dictionary words nor glossary terms",
            example=counts.most_common(1)[0][0],
            count=len(bad),
        )
    ]


def _repeat_findings(transcript: Transcript) -> list[Finding]:
    counts = Counter(line.text for line in transcript.lines)
    repeated = [
        (text, count)
        for text, count in counts.items()
        if count >= _REPEAT_THRESHOLD and len(_tokens(text)) >= _REPEAT_MIN_TOKENS
    ]
    if not repeated:
        return []
    text, count = max(repeated, key=lambda item: item[1])
    return [
        Finding(
            kind="repeat",
            severity="warn",
            detail=f"a {count}-line repetition looks like an OCR loop",
            example=text,
            count=count,
        )
    ]


def _credit_findings(transcript: Transcript) -> list[Finding]:
    """Credit lines that survived into the transcript.

    Judged by ``keep_credit_mask`` -- the reader's own rule, including the
    previous-line shape ("ART BY" / "TURTLEME"). Reusing it rather than
    re-deriving a marker list means a clean transcript reports nothing and any
    hit is a line the reader would itself have dropped.
    """

    texts = [line.text for line in transcript.lines]
    hits = [text for text, keep in zip(texts, keep_credit_mask(texts)) if not keep]
    if not hits:
        return []
    return [
        Finding(
            kind="credit",
            severity="warn",
            detail=f"{len(hits)} line(s) carry credits or a site watermark",
            example=hits[0],
            count=len(hits),
        )
    ]


def _cjk_findings(transcript: Transcript) -> list[Finding]:
    hits = [line.text for line in transcript.lines if has_cjk(line.text)]
    if not hits:
        return []
    return [
        Finding(
            kind="cjk",
            severity="fail",
            detail=f"{len(hits)} line(s) contain hallucinated CJK glyphs",
            example=hits[0],
            count=len(hits),
        )
    ]


def _nonprintable_findings(transcript: Transcript) -> list[Finding]:
    hits = [line.text for line in transcript.lines if _NONPRINTABLE.search(line.text)]
    if not hits:
        return []
    return [
        Finding(
            kind="nonprintable",
            severity="fail",
            detail=f"{len(hits)} line(s) contain control characters",
            example=repr(hits[0])[:80],
            count=len(hits),
        )
    ]


def _page_out_of_range_findings(
    transcript: Transcript, expected_pages: int | None
) -> list[Finding]:
    """Text attributed to a page that is not in the chapter.

    Only this direction is judged. A transcript marks up a page *only when it
    produced text*, so a silent page leaves no marker and a shortfall means
    nothing -- the first version of this check compared marker count to page
    count and reported "7 pages missing" on a complete chapter. A marker that
    exceeds the pages on disk cannot be explained by silence, so it is real.
    """

    if expected_pages is None or not transcript.pages:
        return []
    over = [page for page in transcript.pages if page > expected_pages]
    if not over:
        return []
    return [
        Finding(
            kind="page_out_of_range",
            severity="fail",
            detail=(
                f"{len(over)} page marker(s) exceed the {expected_pages} page(s) "
                f"on disk (highest marker: {max(over)})"
            ),
            example=f"page {max(over)}",
            count=len(over),
        )
    ]


def check_transcript(
    transcript: Transcript, glossary, expected_pages: int | None = None
) -> list[Finding]:
    """Every finding for one transcript, failures first."""

    findings: list[Finding] = []
    findings += _cjk_findings(transcript)
    findings += _nonprintable_findings(transcript)
    findings += _fused_run_findings(transcript, glossary)
    findings += _page_out_of_range_findings(transcript, expected_pages)
    findings += _term_split_findings(transcript, glossary)
    findings += _noise_findings(transcript, glossary)
    findings += _repeat_findings(transcript)
    findings += _credit_findings(transcript)
    return sorted(findings, key=lambda f: (f.severity != "fail", f.kind))


def manifest_page_counts(series_dir) -> dict[int, int]:
    """Ordinal -> pages the manifest says the chapter contains.

    A ``pages`` of 0 means the download stage never recorded a count, so that
    chapter is absent from the map rather than reported as zero pages.
    """

    from comic_manifest import load_manifest

    manifest = load_manifest(Path(series_dir) / "chapters.json")
    return {
        record.ordinal: record.pages
        for record in manifest.chapters
        if record.pages > 0
    }


def check_pages(series_dir) -> list[Finding]:
    """Chapters whose pages on disk fall short of the manifest's count.

    This is the one page defect that is unambiguous and needs no transcript: the
    manifest records what the site said a chapter contained, and the download
    stage records what it actually saved. On the real series ``Chapter 144``
    lists 81 pages and the site answered HTTP 200 with zero bytes for one of
    them, so 80 landed. Nothing downstream can see that -- the transcript is
    merely one page shorter -- which is why it is checked here.

    A chapter with ``pages == 0`` was never downloaded (or was downloaded before
    the count was recorded) and is skipped rather than reported short.
    """

    from comic_manifest import load_manifest

    series_dir = Path(series_dir)
    manifest = load_manifest(series_dir / "chapters.json")
    findings: list[Finding] = []

    for record in manifest.chapters:
        # A detected duplicate (``Chapter 144`` and ``Chapter 144.0`` are the
        # same chapter re-uploaded) is recorded with its page count but
        # deliberately never downloaded, so it has no folder by design.
        # Reporting it as short would flag the duplicate rule working.
        if record.duplicate_of is not None:
            continue
        # ``pages`` is the count the scan recorded from the site listing, so a
        # chapter that was never scanned has 0 and cannot be reported short --
        # ``0 < 0`` is already false. A chapter that *was* scanned and has
        # nothing on disk is a hole, not an unstarted chapter, so there is
        # deliberately no guard for ``on_disk == 0``.
        on_disk = _pages_on_disk(series_dir, record.ordinal)
        if on_disk < record.pages:
            findings.append(
                Finding(
                    kind="page_short",
                    severity="fail",
                    detail=(
                        f"chapter {record.ordinal} ({record.label}) lists "
                        f"{record.pages} page(s) but {on_disk} are on disk"
                    ),
                    example=f"c_{record.ordinal:06d}",
                    count=record.pages - on_disk,
                )
            )
    return findings


def _pages_on_disk(series_dir: Path, ordinal: int) -> int:
    """Page images saved for ``ordinal``, reusing the scraper's own rule.

    ``comic_scraper.pages_on_disk`` already counts both the current
    folder-per-chapter layout and the older flat tree, and is careful to match
    the zero-padded ordinal so chapter 5 cannot absorb chapter 50's pages. It
    imports the reader lazily, so the extra indirection costs nothing here.
    """

    from comic_scraper import pages_on_disk

    return pages_on_disk(series_dir, ordinal)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate comic transcripts without needing the novel."
    )
    parser.add_argument("-n", "--name", required=True,
                        help="slug under novels/, e.g. comic-TBATE")
    parser.add_argument("--config", default=None,
                        help="comic config holding the glossary")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--comics",
        default=None,
        help="series directory holding chapters.json, to check page counts "
        "against the manifest (e.g. comics/TBATE)",
    )
    args = parser.parse_args(argv)

    directory = Path("novels") / args.name
    if not directory.is_dir():
        print(f"no transcripts at {directory}", file=sys.stderr)
        return 2

    glossary = load_glossary(args.config or "src/comic_config.json")
    transcripts = sorted(directory.glob("chapter_*.txt"))
    if not transcripts:
        print(f"no chapter_*.txt in {directory}", file=sys.stderr)
        return 2

    # Without this the page-marker check can never fire from the command line,
    # because nothing hands it a page count. Chapter 147 (81 listed, 80 on disk)
    # is the reason the count matters.
    page_counts = manifest_page_counts(args.comics) if args.comics else {}

    rows = []
    failures = 0
    total_tokens = 0
    total_suspicious = 0

    for path in transcripts:
        transcript = parse_transcript(path.read_text(encoding="utf-8"))
        findings = check_transcript(
            transcript, glossary, expected_pages=page_counts.get(transcript.chapter)
        )
        failures += sum(1 for f in findings if f.severity == "fail")
        total_tokens += len(_tokens(transcript.text))
        total_suspicious += len(suspicious_tokens(transcript, glossary))
        rows.append((path.name, transcript.chapter, findings))

    # Manifest findings are collected before printing so ``--json`` emits one
    # document. Printing them after the JSON text produced a stream no parser
    # accepts, which defeated the point of the flag.
    page_findings = check_pages(args.comics) if args.comics else []
    failures += len(page_findings)

    if args.json:
        print(json.dumps(
            {
                "name": args.name,
                "chapters": len(transcripts),
                "suspicious_rate": (
                    total_suspicious / total_tokens if total_tokens else 0.0
                ),
                "suspicious_tokens": total_suspicious,
                "tokens": total_tokens,
                "failures": failures,
                "findings": [
                    {
                        "file": name,
                        "chapter": chapter,
                        **finding.__dict__,
                    }
                    for name, chapter, findings in rows
                    for finding in findings
                ]
                + [
                    {"file": finding.example, "chapter": None, **finding.__dict__}
                    for finding in page_findings
                ],
            },
            indent=2,
        ))
    else:
        for name, chapter, findings in rows:
            mark = "FAIL" if any(f.severity == "fail" for f in findings) else "ok"
            print(f"{mark:>4}  {name}")
            for finding in findings:
                print(f"        [{finding.severity}] {finding.kind}: {finding.detail}")
                if finding.example:
                    print(f"                 e.g. {finding.example}")
        for finding in page_findings:
            print(f"\nFAIL  {finding.example}")
            print(f"        [{finding.severity}] {finding.kind}: {finding.detail}")
        rate = total_suspicious / total_tokens if total_tokens else 0.0
        print(
            f"\n{len(transcripts)} chapter(s), {failures} failure(s), "
            f"suspicious {total_suspicious}/{total_tokens} tokens ({100 * rate:.1f}%)"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
