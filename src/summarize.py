"""Generate one recap per chapter with the chosen GGUF model.

This is the GGUF counterpart to ``summarizer.py``. It keeps the same
configuration file and the same output tree, but drives ``llama-server``
instead of loading a Hugging Face model into this process, which makes a run
several times faster and lets the model be paused (see "Pause and resume").

Resume is the default: a chapter whose recap already exists is skipped, so
interrupting a run with Ctrl+C and starting the same command again continues
where it stopped. Recaps are written atomically, so an interrupted write can
never leave a half-finished file that would later be mistaken for a real recap.

Examples
--------
    uv run python src/summarize.py -n the-beginning-after-the-end
    uv run python src/summarize.py -n the-beginning-after-the-end -c 1-20
    uv run python src/summarize.py -n the-beginning-after-the-end --overwrite
    uv run python src/summarize.py -n the-beginning-after-the-end --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

from summarizer_gguf import GgufError, LlamaServer
from summary_layers import (
    Piece,
    batch_label,
    pack_groups,
    partial_outputs,
    rounds_needed,
    summary_filename,
)

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "src" / "summarizer_config.json"

#: `chapter_00001.txt` is canonical; `chapter_1.txt` is the pre-migration shape.
CHAPTER_RE = re.compile(r"chapter_0*(\d+)\.txt$", re.IGNORECASE)


class UsageError(RuntimeError):
    """The run was asked to do something impossible; nothing was started."""


# -- configuration ---------------------------------------------------------


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str | None) -> Path | None:
    """Resolve a config path: absolute stays put, relative is repo-root based."""

    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (REPO / path)


def model_entry(config: dict, model_name: str) -> dict:
    """Return a model's config entry, naming the alternatives when unknown.

    ``summarizer.py`` indexes this dict directly, so a typo surfaces as a bare
    ``KeyError``. Listing the known names is more useful than the traceback.
    """

    entries = config["model_config"]
    if model_name not in entries:
        known = ", ".join(sorted(entries))
        raise UsageError(f"Unknown model '{model_name}'. Configured models: {known}")
    return entries[model_name]


def prompt_set(config: dict, entry: dict) -> dict:
    """Return the prompt bundle a model's entry points at.

    Prompts live under one shared ``prompts`` block so every model that opts
    into a set is measured against identical wording; an inline
    ``system_prompt`` on the entry still wins, for backwards compatibility.
    """

    name = entry.get("prompt_set")
    if not name:
        if "system_prompt" in entry or "prompt_template" in entry:
            return entry
        raise UsageError(
            "Model entry defines neither 'prompt_set' nor an inline "
            "'system_prompt'/'prompt_template'"
        )
    sets = config.get("prompts", {})
    if name not in sets:
        known = ", ".join(sorted(sets)) or "none"
        raise UsageError(f"Unknown prompt_set '{name}'. Available: {known}")
    return {**entry, **sets[name]}


def resolve_model_path(config: dict, entry: dict) -> Path:
    """Locate the GGUF file for a model entry."""

    filename = entry.get("gguf_file")
    if not filename:
        raise UsageError(
            "Model entry has no 'gguf_file'. Set it to the .gguf filename "
            "(looked up in MODEL_DIR), or an absolute path."
        )
    candidate = Path(filename)
    if candidate.is_absolute():
        packed = candidate
    else:
        model_dir = resolve_path(config.get("MODEL_DIR"))
        if model_dir is None:
            raise UsageError(
                "Model entry gives a relative 'gguf_file' but MODEL_DIR is unset. "
                "Set MODEL_DIR to the directory holding the .gguf files."
            )
        packed = model_dir / filename
    if not packed.exists():
        raise UsageError(
            f"GGUF model not found: {packed}\n"
            "Download the .gguf into MODEL_DIR, or point 'gguf_file' at it."
        )
    return packed


# -- chapters --------------------------------------------------------------


def discover_chapters(novel_dir: Path) -> dict[int, Path]:
    """Map chapter number to file, sorted, ignoring unparseable filenames."""

    if not novel_dir.is_dir():
        raise UsageError(f"Novel directory not found: {novel_dir}")
    found = {}
    for path in novel_dir.glob("chapter_*.txt"):
        match = CHAPTER_RE.search(path.name)
        if match:
            found[int(match.group(1))] = path
    if not found:
        raise UsageError(f"No 'chapter_*.txt' files in {novel_dir}")
    return dict(sorted(found.items()))


def parse_selection(text: str | None, chapters: dict[int, Path]) -> list[int]:
    """Turn ``-c`` into chapter numbers, supporting ``all`` and ``recent``.

    Same grammar as ``summarizer.py`` (``1``, ``1-5``, ``1,3-5,7``) plus
    ``all`` for the whole book, so existing habits carry over.
    """

    available = sorted(chapters)
    if not text or text.strip().lower() in ("all", "*"):
        return available

    if text.strip().lower() == "recent":
        return [available[-1]]

    selected: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start > end:
                    start, end = end, start
                selected.update(range(start, end + 1))
            else:
                selected.add(int(part))
        except ValueError:
            raise UsageError(f"Invalid chapter selection: '{part}'") from None

    return [number for number in available if number in selected]


# -- output ----------------------------------------------------------------


def summary_path(summaries_dir: Path, chapter: int) -> Path:
    """One recap per chapter, named for the chapter it covers."""

    return summaries_dir / f"summary_chapter_{chapter:05d}.txt"


def write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file so an interrupt cannot leave a partial recap."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def existing_chapters(summaries_dir: Path, numbers: list[int]) -> set[int]:
    """Which of ``numbers`` already have a non-empty recap on disk."""

    done = set()
    for number in numbers:
        path = summary_path(summaries_dir, number)
        if path.is_file() and path.stat().st_size > 0:
            done.add(number)
    return done


def load_recaps(summaries_dir: Path, numbers: list[int]) -> list:
    """Read the recaps for ``numbers`` as mergeable pieces.

    Batch, digest and master steps read these instead of the source chapters,
    so a wider artifact can be rebuilt without re-reading the novel.
    """

    pieces = []
    for number in numbers:
        path = summary_path(summaries_dir, number)
        if path.is_file() and path.stat().st_size > 0:
            pieces.append(Piece(number, number, path.read_text(encoding="utf-8").strip()))
    return pieces


def recapped_chapters(summaries_dir: Path, numbers: list[int]) -> list[int]:
    """Which of ``numbers`` already have a non-empty recap on disk, in order."""

    return [
        number for number in numbers
        if summary_path(summaries_dir, number).is_file()
        and summary_path(summaries_dir, number).stat().st_size > 0
    ]


# -- summary generation ----------------------------------------------------


def prompt_tokens_for(server, prompts: dict, rendered: str) -> int:
    """Exact token cost of one rendered prompt plus its system message."""

    total = server.tokenize(rendered) + 8
    system_prompt = prompts.get("system_prompt")
    if system_prompt:
        total += server.tokenize(system_prompt) + 8
    return total


def render_merge_prompt(prompts: dict, pieces: list, novel: str,
                        target_words: int | None = None) -> str:
    """Render a merge prompt from recaps, labelled so the model can attribute."""

    blocks = []
    for piece in pieces:
        blocks.append(f"{batch_label(piece.start, piece.end)}\n{piece.text}")
    joined = "\n\n".join(blocks)
    return prompts["prompt_template"].format(
        chapter_range=batch_label(pieces[0].start, pieces[-1].end),
        novel=novel,
        summaries=joined,
        chapter=joined,
        target_words=target_words or "",
    )


def segment_for(pieces: list, server, prompts: dict, novel: str,
                n_ctx: int, max_new_tokens: int) -> list[list]:
    """Split pieces into consecutive runs that each fit one prompt.

    Used when a fixed-width batch is too wide for the window. Splitting keeps
    the chapters in order and means a wide batch degrades into a few merges
    rather than failing outright.
    """

    budget = n_ctx - max_new_tokens
    segments: list[list] = []
    current: list = []
    for piece in pieces:
        candidate = [*current, piece]
        if current and prompt_tokens_for(server, prompts,
                                         render_merge_prompt(prompts, candidate, novel)) > budget:
            segments.append(current)
            current = [piece]
        else:
            current = candidate
    if current:
        segments.append(current)

    oversized = [
        seg for seg in segments
        if prompt_tokens_for(server, prompts, render_merge_prompt(prompts, seg, novel)) > budget
    ]
    if oversized:
        raise UsageError(
            f"{batch_label(oversized[0][0].start, oversized[0][-1].end)} does not fit "
            f"in n_ctx={n_ctx} even on its own. Lower 'max_new_tokens' or raise 'n_ctx'."
        )
    return segments


def merge_pieces(server, prompts: dict, pieces: list, novel: str,
                 n_ctx: int, max_new_tokens: int,
                 target_words: int | None = None) -> str:
    """Merge one batch that is already known to fit."""

    rendered = render_merge_prompt(prompts, pieces, novel, target_words)
    messages = []
    system_prompt = prompts.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": rendered})
    return server.chat(messages, max_new_tokens=max_new_tokens)


# -- actions ---------------------------------------------------------------


def summarise_chapters(server, args, config, entry, prompts, chapters,
                       summaries_dir, n_ctx, max_new_tokens) -> int:
    """One recap per chapter. Resumes over whatever already exists."""

    selected = parse_selection(args.chapters, chapters)
    if not selected:
        raise UsageError("Chapter selection matched no chapters.")

    done = set() if args.overwrite else existing_chapters(summaries_dir, selected)
    pending = [number for number in selected if number not in done]

    print(f"Chapters: {len(selected)} selected, {len(done)} already summarised, "
          f"{len(pending)} to do")
    if not pending:
        print("\nNothing to do: every selected chapter already has a recap. "
              "Pass --overwrite to regenerate.")
        return 0

    if args.dry_run:
        preview = ", ".join(str(n) for n in pending[:12])
        more = "" if len(pending) <= 12 else f", ... (+{len(pending) - 12} more)"
        print(f"\n[dry run] would summarise: {preview}{more}")
        return 0

    written = 0
    failures: list[tuple[int, str]] = []
    started = time.time()

    with tqdm(total=len(pending), desc="Summarising") as bar:
        for number in pending:
            bar.set_description(f"Chapter {number}")
            text = chapters[number].read_text(encoding="utf-8")
            try:
                prompt_tokens = budget_check(server, prompts, text, number,
                                             args.novel, n_ctx, max_new_tokens)
                messages = build_messages(prompts, text, number, args.novel)
                chapter_started = time.time()
                recap = server.chat(messages, max_new_tokens=max_new_tokens)
                elapsed = time.time() - chapter_started
            except (GgufError, UsageError) as exc:
                # One bad chapter must not discard an hour of work.
                failures.append((number, str(exc)))
                tqdm.write(f"  chapter {number}: FAILED -- {exc}")
                bar.update(1)
                continue

            write_atomic(summary_path(summaries_dir, number), recap)
            written += 1
            rate = len(recap.split()) / elapsed if elapsed else 0.0
            bar.set_postfix(tokens=f"{prompt_tokens}in", speed=f"{rate:.1f}w/s")
            bar.update(1)

    elapsed = time.time() - started
    print(f"\nWrote {written} recap(s) in {elapsed / 60:.1f} min")
    if written:
        print(f"Average {elapsed / written:.1f}s per chapter")
    if failures:
        print(f"{len(failures)} chapter(s) failed; re-run to retry them:")
        for number, message in failures[:10]:
            print(f"  chapter {number}: {message.splitlines()[0]}")
        return 1
    remaining = [n for n in selected if not summary_path(summaries_dir, n).is_file()]
    if remaining:
        print(f"{len(remaining)} chapter(s) still missing; re-run to continue.")
    return 0


def batches(server, args, config, prompts, numbers, summaries_dir,
            n_ctx, max_new_tokens) -> int:
    """Fixed-width batch summaries, e.g. every 5 chapters."""

    unit = args.group_size
    if unit is None:
        raise UsageError("--group-size is required for batch summaries.")
    pieces = load_recaps(summaries_dir, numbers)

    ranges = partial_outputs(pieces, unit)
    print(f"Batches: {len(ranges)} summary file(s) of {unit} chapter(s) each")
    if args.dry_run:
        pending = 0
        print()
        for start, end in ranges:
            target = summaries_dir / summary_filename(start, end)
            exists = target.is_file() and target.stat().st_size > 0
            done = exists and not args.overwrite
            pending += 0 if done else 1
            block = [p for p in pieces if start <= p.start <= end]
            words = sum(len(p.text.split()) for p in block)
            state = "skip" if done else "write"
            print(f"  {state:5}  {batch_label(start, end):20} "
                  f"({words:,} words in)")
        print(f"\n[dry run] {pending} batch(es) would be written")
        return 0

    written = skipped = 0
    started = time.time()
    for start, end in ranges:
        target = summaries_dir / summary_filename(start, end)
        if target.is_file() and target.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            continue
        block = [p for p in pieces if start <= p.start <= end]
        segments = segment_for(block, server, prompts, args.novel, n_ctx, max_new_tokens)
        if len(segments) == 1:
            text = merge_pieces(server, prompts, segments[0], args.novel,
                                n_ctx, max_new_tokens)
        else:
            # Too wide for one window: merge each run, then merge the results.
            partials = [
                Piece(seg[0].start, seg[-1].end,
                      merge_pieces(server, prompts, seg, args.novel, n_ctx, max_new_tokens))
                for seg in segments
            ]
            text = merge_pieces(server, prompts, partials, args.novel,
                                n_ctx, max_new_tokens)
        write_atomic(target, text)
        written += 1
        tqdm.write(f"  {batch_label(start, end)} -> {len(text.split())} words")

    elapsed = time.time() - started
    print(f"\nWrote {written} batch summar{'y' if written == 1 else 'ies'}"
          f"{f', skipped {skipped} existing' if skipped else ''} "
          f"in {elapsed / 60:.1f} min")
    return 0


def digest_batches(server, prompts: dict, pieces: list, novel: str,
                   n_ctx: int, out_tokens: int, batch_size: int) -> list[list]:
    """Group recaps into digest batches that fit the window.

    Two limits apply at once: ``batch_size`` chapters at most, and no more input
    than the context window leaves room for. Recaps vary in length enough that a
    count alone can overflow -- and an overflowing prompt would silently digest
    only the first chapters of the batch while still writing a plausible file.
    """

    def size_of(text: str) -> int:
        # A dry run has no model, so fall back to a rough chars-per-token guess.
        return server.tokenize(text) if server else max(1, len(text) // 4)

    overhead = size_of(prompts["prompt_template"].format(novel=novel, chapter=""))
    budget = max(1, n_ctx - out_tokens - overhead)

    groups: list[list] = []
    current: list = []
    used = 0
    for piece in pieces:
        cost = size_of(f"Chapter {piece.start}:\n{piece.text}")
        if current and (len(current) >= batch_size or used + cost > budget):
            groups.append(current)
            current, used = [piece], cost
        else:
            current.append(piece)
            used += cost
    if current:
        groups.append(current)
    return groups


def digest(server, args, config, prompts, numbers, summaries_dir,
           n_ctx, max_new_tokens) -> int:
    """One or two sentences per chapter, in a single skim-readable file."""

    settings = config.get("digest", {})
    batch_size = int(settings.get("batch_size", 12))
    filename = settings.get("digest_file", "digest.txt")
    # A digest batch returns one short line per chapter, so it needs far fewer
    # output tokens than a recap does -- but the budget must still cover the
    # whole batch or the last chapters of it get cut off.
    out_tokens = int(settings.get("max_new_tokens", 2048))
    target = summaries_dir / filename

    pieces = load_recaps(summaries_dir, numbers)

    # Existing lines are always read back, even with --overwrite: the flag means
    # "regenerate these chapters", not "discard the chapters I did not select".
    # Dropping them would silently shrink the digest on a narrow -c range.
    lines: dict[int, str] = {}
    if target.is_file():
        for line in target.read_text(encoding="utf-8").splitlines():
            parsed = parse_digest_line(line)
            if parsed:
                lines[parsed[0]] = parsed[1]
    present = sum(1 for p in pieces if p.start in lines)
    print(f"Digest: {present} of {len(pieces)} chapter(s) already present "
          f"in {target.name}")

    todo = [p for p in pieces if args.overwrite or p.start not in lines]
    if not todo:
        print(f"\nNothing to do: {target.name} covers every selected chapter. "
              "Pass --overwrite to rebuild it.")
        return 0

    batches_of = digest_batches(server, prompts, todo, args.novel,
                                n_ctx, out_tokens, batch_size)
    if args.dry_run:
        print(f"\n[dry run] {len(batches_of)} batch(es) would add {len(todo)} "
              f"line(s) to {target.name}")
        for group in batches_of:
            print(f"  Chapters {group[0].start}-{group[-1].start} "
                  f"({len(group)} chapter(s))")
        return 0

    started = time.time()
    with tqdm(total=len(batches_of), desc="Digesting") as bar:
        for group in batches_of:
            bar.set_description(
                f"Chapters {group[0].start}-{group[-1].start}"
            )
            rendered = prompts["prompt_template"].format(
                novel=args.novel,
                chapter="\n\n".join(
                    f"Chapter {p.start}:\n{p.text}" for p in group
                ),
            )
            if prompt_tokens_for(server, prompts, rendered) + out_tokens > n_ctx:
                raise UsageError(
                    f"Digest batch {group[0].start}-{group[-1].start} needs more "
                    f"than n_ctx={n_ctx}. Lower 'digest.batch_size'."
                )
            messages = []
            system_prompt = prompts.get("system_prompt")
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": rendered})
            reply = server.chat(messages, max_new_tokens=out_tokens)
            for line in reply.splitlines():
                parsed = parse_digest_line(line)
                if parsed:
                    lines[parsed[0]] = parsed[1]
            # Rewritten each batch so an interruption keeps finished work.
            write_atomic(target, render_digest(lines))
            bar.update(1)

    print(f"\nWrote {len(lines)} digest line(s) to {target} "
          f"in {(time.time() - started) / 60:.1f} min")
    missing_lines = [p.start for p in pieces if p.start not in lines]
    if missing_lines:
        print(f"{len(missing_lines)} chapter(s) produced no line "
              f"(e.g. {missing_lines[:5]}); re-run to retry them.")
        return 1
    return 0


def budget_check(server, prompts: dict, chapter_text: str, number: int,
                 novel: str, n_ctx: int, max_new_tokens: int) -> int:
    """Refuse to summarise a chapter that cannot fit the context window.

    llama.cpp silently keeps only the first ``n_ctx`` tokens of an oversized
    prompt, which would produce a confident recap of just the opening pages.
    Counting exactly and failing loudly is the only safe option.
    """

    messages = build_messages(prompts, chapter_text, number, novel)
    prompt_tokens = sum(
        server.tokenize(message["content"]) + 8 for message in messages
    )
    if prompt_tokens + max_new_tokens > n_ctx:
        raise UsageError(
            f"Chapter {number} needs {prompt_tokens + max_new_tokens:,} tokens "
            f"({prompt_tokens:,} prompt + {max_new_tokens:,} output) but n_ctx is "
            f"{n_ctx:,}. Raise 'n_ctx' for this model or lower 'max_new_tokens'."
        )
    return prompt_tokens


def parse_digest_line(line: str) -> tuple[int, str] | None:
    """Read one ``Chapter N: text`` digest line, or ``None`` if it is not one.

    Tolerates the markdown emphasis and dash punctuation a model sometimes adds
    (``**Chapter 7:**``, ``Chapter 7 -``) so a stray asterisk does not silently
    drop a chapter from the digest.
    """

    match = re.match(
        r"\**\s*Chapter\s+(\d+)\s*\**\s*[:\u2014.\-]?\s*\**(.*)$",
        line.strip(), re.IGNORECASE,
    )
    if not match:
        return None
    text = match.group(2).strip().lstrip(":-\u2014.").strip()
    if not text:
        return None
    return int(match.group(1)), text


def render_digest(lines: dict[int, str]) -> str:
    """One line per chapter, in chapter order, blank line between each."""

    return "\n\n".join(
        f"Chapter {number}: {lines[number]}" for number in sorted(lines)
    ) + "\n"


def master(server, args, config, prompts, numbers, summaries_dir,
           n_ctx, max_new_tokens) -> int:
    """Whole-novel summary, compressing repeatedly until it fits."""

    settings = config.get("master", {})
    fan_in = int(args.fan_in or settings.get("fan_in", 20))
    target_words = int(settings.get("target_words", 600))
    target = summaries_dir / "master_summary.txt"

    pieces = load_recaps(summaries_dir, numbers)
    novel = args.novel

    if (target.is_file() and target.stat().st_size > 0 and not args.overwrite
            and not args.dry_run):
        print(f"Nothing to do: {target.name} already exists. "
              "Pass --overwrite to rebuild it.")
        return 0

    # A smaller budget for the top of the tree: the last merges must land near
    # target_words, and giving them the recap-sized allowance would just produce
    # another full-length recap of the whole novel.
    final_tokens = max(256, int(target_words * 3.0))
    budget = n_ctx - max_new_tokens

    # Rounds are decided from real token counts, not a guessed tree depth: the
    # first round packs recaps as densely as the window allows, and each later
    # round re-measures whatever the previous one actually produced.
    estimate = rounds_needed(len(pieces), fan_in)

    print(f"Master summary: {len(pieces)} recaps, fan-in {fan_in}, "
          f"~{estimate} merge round(s)")
    if args.dry_run:
        # No model is loaded for a dry run, so this is a word-count estimate
        # rather than the exact measurement the real run uses. Enough to show
        # the shape of the tree.
        words = sum(len(p.text.split()) for p in pieces)
        approx_tokens = max(1, words * 4 // 3)
        first = min(fan_in, max(1, budget * len(pieces) // approx_tokens))
        print(f"  round 1: ~{-(-len(pieces) // first)} merge(s) "
              f"of up to {first} recap(s)")
        print(f"\n[dry run] would write {target.name} (~{target_words} words) "
              f"from {words:,} words of recaps")
        return 0

    def fits(candidate: list) -> bool:
        rendered = render_merge_prompt(prompts, candidate, novel)
        return prompt_tokens_for(server, prompts, rendered) <= budget

    def fits_final(candidate: list) -> bool:
        rendered = render_merge_prompt(prompts, candidate, novel)
        return prompt_tokens_for(server, prompts, rendered) <= n_ctx - final_tokens

    started = time.time()
    current = pieces
    level = 0
    truncated = 0
    while len(current) > 1:
        level += 1
        groups = pack_groups(current, fan_in=fan_in, fits=fits)
        if len(groups) >= len(current):
            raise UsageError(
                f"Cannot compress further: {len(current)} pieces still produce "
                f"{len(groups)} prompt(s) at n_ctx={n_ctx}. Raise 'n_ctx' or lower "
                "'master.fan_in'."
            )
        last_level = len(groups) == 1
        produced: list[Piece] = []
        with tqdm(total=len(groups), desc=f"Round {level}") as bar:
            for group in groups:
                limit = final_tokens if last_level else max_new_tokens
                # The final round must also fit under its smaller allowance.
                if last_level and not fits_final(group):
                    raise UsageError(
                        f"The last merge ({batch_label(group[0].start, group[-1].end)}) "
                        f"needs more than {n_ctx - final_tokens:,} prompt tokens. "
                        "Raise 'n_ctx' or lower 'master.target_words'."
                    )
                text = merge_pieces(server, prompts, group, novel, n_ctx, limit,
                                    target_words)
                if getattr(server, "last_finish_reason", None) == "length":
                    truncated += 1
                produced.append(Piece(group[0].start, group[-1].end, text))
                bar.update(1)
        current = produced
        # Checkpoint after every round, so an interrupted run still leaves a
        # usable summary of the whole book on disk.
        write_atomic(target, render_master(current))

    write_atomic(target, render_master(current))
    words = len(current[0].text.split()) if current else 0
    print(f"\nWrote {target.name} ({words} words, {level} round(s)) "
          f"in {(time.time() - started) / 60:.1f} min")
    if truncated:
        print(f"{truncated} merge(s) hit the output cap and may end mid-sentence. "
              "Raise 'master.target_words' and re-run with --overwrite for more.")
    return 0


def render_master(pieces: list) -> str:
    """Assemble the master summary file.

    When the tree has not collapsed to a single node -- an interrupted run, or
    a corpus small enough that one round covers it -- every node is written out
    rather than silently discarding all but the first.
    """

    body = "\n\n".join(piece.text.strip() for piece in pieces if piece.text.strip())
    header = "# Novel summary\n\n"
    if len(pieces) > 1:
        header += ("_Partial summary: these sections cover "
                   + ", ".join(piece.span for piece in pieces)
                   + "._\n\n")
    return header + body + "\n"


def build_messages(prompts: dict, chapter_text: str, number: int, novel: str) -> list[dict]:
    """Assemble the chat turns for one chapter."""

    template = prompts["prompt_template"]
    prompt = template.format(
        chapter=chapter_text,
        chapter_range=f"Chapter {number}",
        novel=novel,
    )
    messages = []
    system_prompt = prompts.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def resolve_action_prompts(config: dict, set_name: str) -> dict:
    """Resolve a prompt set by name, for the batch/digest/master steps."""

    sets = config.get("prompts", {})
    if set_name not in sets:
        known = ", ".join(sorted(sets)) or "none"
        raise UsageError(f"Unknown prompt_set '{set_name}'. Available: {known}")
    return dict(sets[set_name])


def run(args) -> int:
    config = load_config()
    entry = prompt_set(config, model_entry(config, args.model))

    if entry.get("backend", "transformers") != "gguf":
        raise UsageError(
            f"Model '{args.model}' is configured for the '{entry.get('backend')}' "
            "backend. This script runs GGUF models only; use src/summarizer.py "
            "for the transformers path."
        )

    model_path = resolve_model_path(config, entry)
    binary = resolve_path(config.get("LLAMA_BINARY"))

    novel_dir = resolve_path(config["NOVEL_DIR"]) / args.novel
    summaries_dir = resolve_path(config["save_dir"]) / args.novel
    chapters = discover_chapters(novel_dir)

    n_ctx = int(entry.get("n_ctx", 12288))
    generation = entry.get("generation", {})
    max_new_tokens = int(generation.get("max_new_tokens", 1400))
    temperature = float(generation.get("temperature", 0.0))

    # Which step(s) to run. Each reads the recaps written by the previous one,
    # so asking for several at once runs them in dependency order.
    actions: list[str] = []
    if args.all:
        actions.append("master")
    if args.digest:
        actions.append("digest")
    if args.group_size is not None:
        actions.append("batches")
    if not actions:
        actions.append("chapters")

    print(f"Novel:   {args.novel}")
    print(f"Model:   {args.model}  ({model_path.name})")
    print(f"Output:  {summaries_dir}")

    # The chapter selection applies to every step, so '-c 1-20 --digest' means
    # a digest of those twenty chapters rather than the whole file.
    selected = parse_selection(args.chapters, chapters)
    if not selected:
        raise UsageError("Chapter selection matched no chapters.")

    covered = selected
    if "chapters" not in actions:
        # Wider artifacts are built from recaps. A hole in the input would
        # produce a batch or digest that looks complete and is wrong, so a
        # missing recap is an error rather than something to skip past.
        have = set(recapped_chapters(summaries_dir, selected))
        missing = [n for n in selected if n not in have]
        if missing:
            preview = ", ".join(str(n) for n in missing[:8])
            more = "" if len(missing) <= 8 else f", ... (+{len(missing) - 8} more)"
            raise UsageError(
                f"{len(missing)} selected chapter(s) have no recap yet: "
                f"{preview}{more}. Run the per-chapter step for them first."
            )

    prompt_for = {
        "chapters": entry,
        "batches": resolve_action_prompts(config, "merge"),
        "digest": resolve_action_prompts(config, "digest"),
        "master": resolve_action_prompts(config, "master"),
    }

    # Free pre-flight check: everything skipped means no need to load a model.
    if actions == ["chapters"]:
        done = set() if args.overwrite else existing_chapters(summaries_dir, covered)
        if len(done) == len(covered) and not args.dry_run:
            print(f"\nNothing to do: all {len(covered)} selected chapter(s) already "
                  "have a recap. Pass --overwrite to regenerate.")
            return 0

    # A dry run only reports what would happen, and every estimate it needs is
    # computable without the model. Loading 5 GB of weights to then print a plan
    # would be both slow and pointless.
    if args.dry_run:
        status = 0
        for action in actions:
            print()
            if action == "chapters":
                status |= summarise_chapters(None, args, config, entry,
                                             prompt_for["chapters"], chapters,
                                             summaries_dir, n_ctx, max_new_tokens)
            elif action == "batches":
                status |= batches(None, args, config, prompt_for["batches"],
                                  covered, summaries_dir, n_ctx, max_new_tokens)
            elif action == "digest":
                status |= digest(None, args, config, prompt_for["digest"],
                                 covered, summaries_dir, n_ctx, max_new_tokens)
            elif action == "master":
                status |= master(None, args, config, prompt_for["master"],
                                 covered, summaries_dir, n_ctx, max_new_tokens)
        return status

    server = LlamaServer(
        binary,
        model_path,
        n_ctx=n_ctx,
        n_gpu_layers=int(entry.get("n_gpu_layers", 999)),
        temperature=temperature,
        reasoning=bool(entry.get("reasoning", False)),
        startup_timeout=float(entry.get("startup_timeout", 600)),
        request_timeout=float(entry.get("request_timeout", 1800)),
    )

    print(f"\nStarting llama-server ({model_path.name}, n_ctx={n_ctx}) ...", flush=True)
    started = time.time()
    server.start()
    print(f"Model ready in {time.time() - started:.1f}s", flush=True)

    status = 0
    try:
        for action in actions:
            print()
            if action == "chapters":
                status |= summarise_chapters(server, args, config, entry,
                                             prompt_for["chapters"], chapters,
                                             summaries_dir, n_ctx, max_new_tokens)
            elif action == "batches":
                status |= batches(server, args, config, prompt_for["batches"],
                                  covered, summaries_dir,
                                  n_ctx, max_new_tokens)
            elif action == "digest":
                status |= digest(server, args, config, prompt_for["digest"],
                                 covered, summaries_dir,
                                 n_ctx, max_new_tokens)
            elif action == "master":
                status |= master(server, args, config, prompt_for["master"],
                                 covered, summaries_dir,
                                 n_ctx, max_new_tokens)
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        status |= 130
    finally:
        # Terminating the child tears down its CUDA context, so the VRAM is
        # actually returned to the system while the run is paused.
        server.stop()
        print("Model unloaded; VRAM released.", flush=True)

    return status


def parse_args(argv: list[str] | None = None):
    config = load_config()
    parser = argparse.ArgumentParser(
        description=(
            "Summarise a novel with a GGUF model (llama.cpp). By default writes one "
            "recap per chapter; --group-size, --digest and --all build wider "
            "artifacts from those recaps."
        )
    )
    parser.add_argument("-n", "--novel", default=config.get("novel"),
                        help="Novel directory name under novels/ (required)")
    parser.add_argument("-m", "--model", default=config.get("model"),
                        help=f"Model key from summarizer_config.json (default: {config.get('model')})")
    parser.add_argument("-c", "--chapters", default=config.get("chapters", "all"),
                        help="Chapters: 'all', 'recent', '5', '1-20', '1,3-5'")
    parser.add_argument("-g", "--group-size", type=int, default=None, metavar="N",
                        help="Also write one summary per N chapters, built from the recaps")
    parser.add_argument("--fan-in", type=int, default=None, metavar="N",
                        help="Chapters merged per step in the --all tree "
                             "(default: the master.fan_in config value)")
    parser.add_argument("--digest", action="store_true",
                        help="Also write digest.txt: one or two sentences per chapter")
    parser.add_argument("--all", "--master", dest="all", action="store_true",
                        help="Also write a whole-novel summary, compressing in rounds "
                             "until it fits")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate artifacts that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be produced, then stop")
    args = parser.parse_args(argv)
    if not args.novel:
        parser.error("no novel given; pass -n <novel-directory-name>")
    if args.group_size is not None and args.group_size < 1:
        parser.error("--group-size must be at least 1")
    if args.fan_in is not None and args.fan_in < 2:
        # A fan-in of 1 would merge nothing, so the tree could never collapse.
        parser.error("--fan-in must be at least 2")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
