"""Tests for the GGUF runner: chapter selection, resume, and context budgeting.

The runner is deliberately thin, but the parts that decide *what gets written*
are worth pinning down: a chapter that silently fails to be included, or one
whose prompt overflows the context window, produces a plausible-looking but
wrong summary tree.
"""

import json
import time

import pytest

from summarize import (
    UsageError,
    build_messages,
    discover_chapters,
    existing_chapters,
    model_entry,
    parse_selection,
    prompt_set,
    resolve_path,
    summary_path,
    write_atomic,
)
from summarizer_gguf import GgufError, LlamaServer
from summary_layers import Piece


# --------------------------------------------------------------------------
# chapter discovery and selection
# --------------------------------------------------------------------------

def test_discover_chapters_reads_padded_names(tmp_path):
    for name in ("chapter_00001.txt", "chapter_00002.txt"):
        (tmp_path / name).write_text("text", encoding="utf-8")
    chapters = discover_chapters(tmp_path)

    assert sorted(chapters) == [1, 2]
    assert chapters[1].name == "chapter_00001.txt"


def test_discover_chapters_still_reads_legacy_unpadded_names(tmp_path):
    """Pre-migration files are deprecated, not forbidden."""
    (tmp_path / "chapter_7.txt").write_text("text", encoding="utf-8")
    assert sorted(discover_chapters(tmp_path)) == [7]


def test_discover_chapters_ignores_unrelated_files(tmp_path):
    (tmp_path / "chapter_00003.txt").write_text("text", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    (tmp_path / "chapter_00003.txt.bak").write_text("text", encoding="utf-8")
    assert sorted(discover_chapters(tmp_path)) == [3]


def test_discover_chapters_reports_an_empty_directory(tmp_path):
    with pytest.raises(UsageError) as err:
        discover_chapters(tmp_path)
    assert "chapter_" in str(err.value)


def test_parse_selection_all_is_the_whole_book():
    chapters = {n: None for n in range(1, 517)}
    assert parse_selection("all", chapters) == list(range(1, 517))
    assert parse_selection(None, chapters) == list(range(1, 517))


def test_parse_selection_handles_ranges_lists_and_combinations():
    chapters = {n: None for n in range(1, 21)}
    assert parse_selection("5", chapters) == [5]
    assert parse_selection("1-3", chapters) == [1, 2, 3]
    assert parse_selection("1-3,7,9-10", chapters) == [1, 2, 3, 7, 9, 10]


def test_parse_selection_recent_is_the_highest_chapter_present():
    """'recent' must follow the files on disk, not a hardcoded count."""
    chapters = {1: None, 2: None, 9: None}
    assert parse_selection("recent", chapters) == [9]


def test_parse_selection_skips_chapters_that_do_not_exist():
    """A gap in the corpus is not an error; the rest still summarises."""
    chapters = {1: None, 3: None}
    assert parse_selection("1-3", chapters) == [1, 3]


def test_parse_selection_rejects_nonsense():
    with pytest.raises(UsageError):
        parse_selection("one-three", {1: None})


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def test_model_entry_names_the_alternatives_when_unknown():
    config = {"model_config": {"a": {}, "b": {}}}
    with pytest.raises(UsageError) as err:
        model_entry(config, "typo")
    message = str(err.value)
    assert "typo" in message
    assert "a, b" in message


def test_prompt_set_is_shared_so_models_cannot_drift():
    config = {
        "prompts": {"recap_faithful": {"system_prompt": "shared", "prompt_template": "{chapter}"}},
        "model_config": {"m": {"prompt_set": "recap_faithful", "chunk_size": 1}},
    }
    entry = prompt_set(config, config["model_config"]["m"])

    assert entry["system_prompt"] == "shared"
    assert entry["prompt_template"] == "{chapter}"
    # per-model settings survive the merge
    assert entry["chunk_size"] == 1


def test_prompt_set_reports_a_missing_prompt_set():
    config = {"prompts": {}, "model_config": {"m": {"prompt_set": "nope"}}}
    with pytest.raises(UsageError) as err:
        prompt_set(config, config["model_config"]["m"])
    assert "nope" in str(err.value)


def test_prompt_set_accepts_an_inline_prompt_for_old_entries():
    entry = {"system_prompt": "inline", "prompt_template": "{chapter}"}
    assert prompt_set({"prompts": {}}, entry)["system_prompt"] == "inline"


def test_resolve_path_anchors_relative_paths_at_the_repo_root():
    resolved = resolve_path("novels")
    assert resolved.is_absolute()
    assert resolved.name == "novels"


def test_resolve_path_leaves_absolute_paths_alone():
    assert str(resolve_path("S:/Programming/models")) == str(
        __import__("pathlib").Path("S:/Programming/models")
    )


# --------------------------------------------------------------------------
# output and resume
# --------------------------------------------------------------------------

def test_summary_path_is_padded_and_per_chapter(tmp_path):
    assert summary_path(tmp_path, 7).name == "summary_chapter_00007.txt"


def test_write_atomic_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "summary_chapter_00001.txt"
    write_atomic(target, "a recap")

    assert target.read_text(encoding="utf-8") == "a recap"
    assert list(tmp_path.iterdir()) == [target]


def test_write_atomic_uses_lf_endings(tmp_path):
    """Output must match the corpus convention, not the host's CRLF default."""
    target = tmp_path / "s.txt"
    write_atomic(target, "one\ntwo\n")
    assert target.read_bytes() == b"one\ntwo\n"


def test_write_atomic_replaces_an_existing_recap(tmp_path):
    target = tmp_path / "s.txt"
    target.write_text("old", encoding="utf-8")
    write_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_existing_chapters_drives_resume(tmp_path):
    write_atomic(summary_path(tmp_path, 1), "done")
    write_atomic(summary_path(tmp_path, 3), "done")

    assert existing_chapters(tmp_path, [1, 2, 3]) == {1, 3}


def test_existing_chapters_ignores_empty_files(tmp_path):
    """An interrupted write must not count as a finished chapter."""
    summary_path(tmp_path, 1).write_text("", encoding="utf-8")
    assert existing_chapters(tmp_path, [1]) == set()


# --------------------------------------------------------------------------
# prompt assembly and context budget
# --------------------------------------------------------------------------

PROMPTS = {
    "system_prompt": "Recap faithfully.",
    "prompt_template": "Summarize {chapter_range} of \"{novel}\".\n\n### Excerpt:\n{chapter}\n\n### Recap:",
}

DIGEST_PROMPTS = {
    "system_prompt": "One line per chapter.",
    "prompt_template": "Digest \"{novel}\".\n\n{chapter}\n\nDigest:",
}


def test_build_messages_puts_the_system_prompt_first():
    messages = build_messages(PROMPTS, "CHAPTER BODY", 12, "a-novel")

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "Recap faithfully."
    assert "Chapter 12" in messages[1]["content"]
    assert "CHAPTER BODY" in messages[1]["content"]
    assert "a-novel" in messages[1]["content"]


def test_build_messages_omits_an_absent_system_prompt():
    prompts = {k: v for k, v in PROMPTS.items() if k != "system_prompt"}
    assert [m["role"] for m in build_messages(prompts, "x", 1, "n")] == ["user"]


class CountingServer:
    """Stands in for LlamaServer.tokenize without a real model."""

    def __init__(self, per_call):
        self.per_call = per_call

    def tokenize(self, text):
        return self.per_call


def test_budget_check_refuses_a_chapter_that_would_be_truncated():
    """Overflow must fail loudly: llama.cpp would silently recap only the head."""
    from summarize import budget_check

    server = CountingServer(12_000)
    with pytest.raises(UsageError) as err:
        budget_check(server, PROMPTS, "body", 400, "n", n_ctx=12_288,
                     max_new_tokens=1_400)
    message = str(err.value)
    assert "n_ctx" in message
    assert "Chapter 400" in message


def test_budget_check_allows_a_chapter_that_fits():
    from summarize import budget_check

    server = CountingServer(100)
    tokens = budget_check(server, PROMPTS, "body", 1, "n", n_ctx=12_288,
                          max_new_tokens=1_400)
    assert tokens == 2 * (100 + 8)


# --------------------------------------------------------------------------
# digest batching
# --------------------------------------------------------------------------

def test_digest_batches_splits_on_the_context_budget_not_just_the_count():
    """Recap lengths vary; a count-only batch can overflow and lose chapters."""
    from summarize import digest_batches

    pieces = [Piece(n, n, "word " * 100) for n in range(1, 11)]
    # 120 tokens each: only two fit any sensible budget.
    server = CountingServer(120)
    groups = digest_batches(server, DIGEST_PROMPTS, pieces, "n",
                            n_ctx=400, out_tokens=100, batch_size=10)
    assert sum(len(g) for g in groups) == 10
    for group in groups:
        assert 1 <= len(group) <= 2


def test_digest_batches_respects_the_chapter_cap():
    from summarize import digest_batches

    pieces = [Piece(n, n, "short") for n in range(1, 11)]
    server = CountingServer(5)
    groups = digest_batches(server, DIGEST_PROMPTS, pieces, "n",
                            n_ctx=12_288, out_tokens=100, batch_size=3)
    assert [len(g) for g in groups] == [3, 3, 3, 1]


def test_digest_batches_never_drops_a_chapter():
    from summarize import digest_batches

    pieces = [Piece(n, n, "word " * 50) for n in range(1, 6)]
    server = CountingServer(400)
    groups = digest_batches(server, DIGEST_PROMPTS, pieces, "n",
                            n_ctx=500, out_tokens=100, batch_size=10)
    listed = [p.start for group in groups for p in group]
    assert listed == [1, 2, 3, 4, 5]


def test_digest_overwrite_keeps_chapters_outside_the_selection(tmp_path):
    """--overwrite on -c 1 should regenerate chapter 1, not delete 2 and 3."""
    from summarize import digest

    class EchoServer:
        def tokenize(self, text):
            return 10

        def chat(self, messages, max_new_tokens=None):
            return "Chapter 1: rewritten line."

    for number in (1, 2, 3):
        (tmp_path / f"summary_chapter_{number:05d}.txt").write_text(
            f"recap {number}", encoding="utf-8"
        )
    digest_file = tmp_path / "digest.txt"
    digest_file.write_text(
        "Chapter 1: old line.\n\nChapter 2: kept line.\n\nChapter 3: kept too.\n",
        encoding="utf-8",
    )

    class Args:
        overwrite = True
        dry_run = False
        novel = "n"

    prompts = {"prompt_template": "Digest {novel}\n{chapter}"}
    status = digest(EchoServer(), Args(), {"digest": {"batch_size": 5}},
                    prompts, [1], tmp_path, n_ctx=12_288, max_new_tokens=100)

    text = digest_file.read_text(encoding="utf-8")
    assert status == 0
    assert "rewritten line" in text
    # The chapters that were not selected must survive untouched.
    assert "kept line" in text
    assert "kept too" in text


def test_a_failing_chapter_is_reported_without_losing_the_rest(tmp_path):
    """One bad chapter must not raise NameError and abort the whole run."""
    from summarize import summarise_chapters

    class ExplodingServer:
        def tokenize(self, text):
            return 10

        def chat(self, messages, max_new_tokens=None):
            raise GgufError("model exploded")

    class Args:
        chapters = "all"
        overwrite = False
        dry_run = False
        novel = "n"

    chapters = {}
    for number in (1, 2):
        path = tmp_path / f"chapter_{number:05d}.txt"
        path.write_text("body", encoding="utf-8")
        chapters[number] = path

    status = summarise_chapters(ExplodingServer(), Args(), {}, PROMPTS, PROMPTS,
                                chapters, tmp_path, n_ctx=12_288,
                                max_new_tokens=100)

    assert status == 1
    # Neither chapter was written, but the run completed instead of crashing.
    assert not summary_path(tmp_path, 1).exists()
    assert not summary_path(tmp_path, 2).exists()


# --------------------------------------------------------------------------
# recovering from a server that dies mid-run
#
# The 8 GB card does not always survive a long run: a real 467-chapter run
# died at chapter 270 with "CUDA error: out of memory". The server was gone,
# yet the loop went on and logged every remaining chapter as FAILED, so one
# crash looked like 244 broken chapters. These pin down the two halves of the
# fix: restart a dead server, and refuse to keep going if it will not come back.
# --------------------------------------------------------------------------


class AllArgs:
    """Args that select every chapter, so a restart can be seen mid-run."""

    chapters = "all"
    overwrite = False
    dry_run = False
    novel = "n"


class DyingServer:
    """Answers until chapter ``dies_at``, then behaves like a dead server."""

    def __init__(self, dies_at=2):
        self.dies_at = dies_at
        self.starts = 0
        self.chatted = 0
        self.down = False

    def start(self):
        self.starts += 1
        self.down = False
        return self

    def alive(self):
        # A crashed child stays dead until start() is called again; that is
        # what process.poll() reports to the real server.
        return self.starts > 0 and not self.down

    def tokenize(self, text):
        return 10

    def chat(self, messages, max_new_tokens=None):
        self.chatted += 1
        if self.chatted >= self.dies_at:
            # The exact shape of the real failure: the child is gone by the
            # time the client gives up.
            self.down = True
            raise GgufError("connection closed")
        return "a recap"


def _chapter_dir(tmp_path, numbers):
    chapters = {}
    for number in numbers:
        path = tmp_path / f"chapter_{number:05d}.txt"
        path.write_text("body", encoding="utf-8")
        chapters[number] = path
    return chapters


def test_a_dead_server_is_restarted_and_the_run_carries_on(tmp_path):
    """A crashed llama-server is restartable, so the chapters after it are not lost."""

    from summarize import summarise_chapters

    server = DyingServer(dies_at=2)
    chapters = _chapter_dir(tmp_path, (1, 2, 3))

    # First chat fails, so the run must notice and call start() again. From
    # then on the server is healthy, exactly as after a successful reload.
    def start():
        server.starts += 1
        server.down = False
        if server.starts > 1:
            server.dies_at = 99
        return server

    server.start = start

    status = summarise_chapters(server, AllArgs(), {}, PROMPTS, PROMPTS, chapters,
                                tmp_path, n_ctx=12_288, max_new_tokens=100)

    assert server.starts == 2, "the crashed server was never restarted"
    assert summary_path(tmp_path, 1).exists()
    assert summary_path(tmp_path, 3).exists(), "chapters after the restart were dropped"
    assert status == 1, "the chapter that actually failed should still be reported"


def test_a_server_that_cannot_be_restarted_stops_the_run(tmp_path):
    """If the server will not come back, stop -- do not label every chapter failed."""

    from summarize import summarise_chapters

    server = DyingServer(dies_at=1)

    def refuse():
        raise GgufError("CUDA error: out of memory")

    server.start = refuse

    chapters = _chapter_dir(tmp_path, range(1, 6))

    with pytest.raises(GgufError) as err:
        summarise_chapters(server, AllArgs(), {}, PROMPTS, PROMPTS, chapters,
                           tmp_path, n_ctx=12_288, max_new_tokens=100)

    # The message has to say why, because "not running" told us nothing.
    assert "out of memory" in str(err.value).lower()


# --------------------------------------------------------------------------
# tokenize endpoint
# --------------------------------------------------------------------------

def _stub_server(tmp_path):
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"stub")
    return LlamaServer(binary, model, startup_timeout=5)


def test_tokenize_requires_start(tmp_path):
    with pytest.raises(GgufError) as err:
        _stub_server(tmp_path).tokenize("hi")
    assert "not running" in str(err.value).lower()


def test_tokenize_counts_tokens_from_the_server(tmp_path, monkeypatch):
    import subprocess

    from test_summarizer_gguf import FakeProcess, FakeResponse

    seen = {}

    class Req:
        def __init__(self, url, data=None, headers=None, method=None):
            self.full_url = url
            self.data = data

    def fake_urlopen(url_or_req, timeout=None):
        if not hasattr(url_or_req, "data"):
            return FakeResponse({"status": "ok"})
        seen["url"] = url_or_req.full_url
        seen["body"] = json.loads(url_or_req.data.decode("utf-8"))
        return FakeResponse({"tokens": [1, 2, 3, 4, 5]})

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("urllib.request.Request", Req)

    server = _stub_server(tmp_path)
    server.start()
    assert server.tokenize("some text") == 5
    assert seen["url"].endswith("/tokenize")
    assert seen["body"]["content"] == "some text"


def test_tokenize_rejects_an_unexpected_response(tmp_path, monkeypatch):
    import subprocess

    from test_summarizer_gguf import FakeProcess, FakeResponse

    class Req:
        def __init__(self, url, data=None, headers=None, method=None):
            self.full_url = url
            self.data = data

    def fake_urlopen(url_or_req, timeout=None):
        if not hasattr(url_or_req, "data"):
            return FakeResponse({"status": "ok"})
        return FakeResponse({"unexpected": True})

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("urllib.request.Request", Req)

    server = _stub_server(tmp_path)
    server.start()
    with pytest.raises(GgufError) as err:
        server.tokenize("text")
    assert "tokenize" in str(err.value).lower()


# --------------------------------------------------------------------------
# crash forensics
#
# A stalled generation used to look like a working run: the request waited out
# its full timeout (1800s) and then reported a generic "did not respond". The
# server's own state was never sampled, so a GPU eviction and a wedged lock were
# indistinguishable. These pin down what a stall must report instead.
# --------------------------------------------------------------------------

def test_a_stall_names_the_slot_wait_and_the_last_log_line(tmp_path, monkeypatch):
    """A dead server must say what it was doing, not just 'no response'."""

    import urllib.error

    from test_summarizer_gguf import FakeProcess, FakeResponse

    def fake_urlopen(url_or_req, timeout=None):
        if not hasattr(url_or_req, "data"):
            return FakeResponse({"status": "ok"})
        raise urllib.error.URLError("timed out")

    server = _stub_server(tmp_path)
    monkeypatch.setattr(server, "vram_report", lambda: "")
    monkeypatch.setattr(server, "_spawn", lambda cmd, env: FakeProcess())
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    server.start()
    # The server logged something before it wedged; that line is the only clue.
    server.log_path.write_text(
        "0.25.763.928 I slot print_timing: id 0 | eval time = 14495.28 ms\n"
        "0.25.768.276 I slot release: id 0 | stop processing: n_tokens = 5847\n",
        encoding="utf-8",
    )

    with pytest.raises(GgufError) as err:
        server.chat([{"role": "user", "content": "x"}])

    message = str(err.value)
    assert "did not respond" in message
    # The last thing the server managed to log must be quoted back.
    assert "stop processing" in message
    assert "llama-server-" in message or ".log" in message


def test_stall_detection_reports_vram_when_a_sampler_is_supplied(tmp_path, monkeypatch):
    """VRAM pressure is the leading suspect for an 8GB card, so measure it."""

    import urllib.error

    from test_summarizer_gguf import FakeProcess, FakeResponse

    def fake_urlopen(url_or_req, timeout=None):
        if not hasattr(url_or_req, "data"):
            return FakeResponse({"status": "ok"})
        raise urllib.error.URLError("timed out")

    server = _stub_server(tmp_path)
    monkeypatch.setattr(server, "_spawn", lambda cmd, env: FakeProcess())
    monkeypatch.setattr(server, "vram_report", lambda: "GPU 7980/8192 MiB")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    server.start()

    with pytest.raises(GgufError) as err:
        server.chat([{"role": "user", "content": "x"}])

    assert "7980/8192 MiB" in str(err.value)


def test_a_stall_reports_a_server_that_dies_mid_request_as_dead(tmp_path, monkeypatch):
    """The socket closes slightly before the process handle reports exit.

    Observed for real: `taskkill /F` produced "server alive: yes" because
    ``poll()`` had not caught up yet. A forensics line that lies is worse than
    no line at all.
    """

    import urllib.error

    from test_summarizer_gguf import FakeProcess, FakeResponse

    def fake_urlopen(url_or_req, timeout=None):
        if not hasattr(url_or_req, "data"):
            return FakeResponse({"status": "ok"})
        raise urllib.error.URLError("[WinError 10054] connection forcibly closed")

    server = _stub_server(tmp_path)
    monkeypatch.setattr(server, "vram_report", lambda: "")

    class DyingProcess(FakeProcess):
        """Alive at first, dead once Windows has had time to signal the handle."""

        def __init__(self):
            super().__init__()
            self.failed_at = None

        def poll(self):
            if self.failed_at is None:
                return None
            if time.time() - self.failed_at < 0.2:
                return None
            return 1

    proc = DyingProcess()

    def killing_urlopen(url_or_req, timeout=None):
        if not hasattr(url_or_req, "data"):
            return FakeResponse({"status": "ok"})
        proc.failed_at = time.time()
        raise urllib.error.URLError("[WinError 10054] connection forcibly closed")

    server = _stub_server(tmp_path)
    monkeypatch.setattr(server, "vram_report", lambda: "")
    monkeypatch.setattr(server, "_spawn", lambda cmd, env: proc)
    monkeypatch.setattr("urllib.request.urlopen", killing_urlopen)

    server.start()
    with pytest.raises(GgufError) as err:
        server.chat([{"role": "user", "content": "x"}])

    assert "server alive: no" in str(err.value)


def test_health_reports_whether_the_server_is_alive(tmp_path, monkeypatch):

    import subprocess

    from test_summarizer_gguf import FakeProcess, FakeResponse

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: FakeResponse({}))

    server = _stub_server(tmp_path)
    server.start()
    assert server.alive() is True

    server.process.returncode = 1  # died under us
    assert server.alive() is False


# --------------------------------------------------------------------------
# starting must not leak a server
#
# ``server.start()`` was called outside the try/finally that unloads the model,
# so Ctrl+C during the ~9s model load left llama-server alive holding ~6.5GB.
# A later run then competed with the orphan for VRAM.
# --------------------------------------------------------------------------

class Args:
    novel = "the-beginning-after-the-end"
    chapters = "1"
    overwrite = False
    dry_run = False
    all = False
    digest = False
    group_size = None
    fan_in = None
    model = "qwen3.5-9b"


def test_interrupt_during_startup_still_unloads_the_server(tmp_path, monkeypatch):
    """Ctrl+C during the ~9s model load must not leave llama-server resident."""

    from test_summarizer_gguf import FakeProcess

    import summarize

    proc = FakeProcess()
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: proc)

    (tmp_path / "llama-server.exe").write_bytes(b"stub")
    (tmp_path / "m.gguf").write_bytes(b"stub")

    # Spawn for real, then interrupt while waiting for /health -- the exact
    # moment the model is loading and most of the VRAM is already committed.
    def interrupt_while_waiting(self):
        self.process = self._spawn(["stub"], {})
        raise KeyboardInterrupt

    monkeypatch.setattr(LlamaServer, "start", interrupt_while_waiting)
    monkeypatch.setattr(summarize, "discover_chapters",
                        lambda d: {1: tmp_path / "chapter_00001.txt"})
    (tmp_path / "chapter_00001.txt").write_text("body", encoding="utf-8")

    monkeypatch.setattr(summarize, "load_config", lambda path=None: config_for(tmp_path))
    monkeypatch.setattr(summarize, "resolve_model_path",
                        lambda c, e: tmp_path / "m.gguf")

    args = Args()
    args.novel = "n"
    status = summarize.run(args)

    assert status == 130
    assert proc.terminated or proc.killed


def config_for(tmp_path):
    """A minimal config whose paths all resolve inside ``tmp_path``."""

    return {
        "NOVEL_DIR": str(tmp_path),
        "save_dir": str(tmp_path / "out"),
        "MODEL_DIR": str(tmp_path),
        "LLAMA_BINARY": str(tmp_path / "llama-server.exe"),
        "model": "qwen3.5-9b",
        "chapters": "1",
        "prompts": {
            "recap_faithful": PROMPTS,
            "merge": PROMPTS,
            "digest": DIGEST_PROMPTS,
            "master": PROMPTS,
        },
        "model_config": {
            "qwen3.5-9b": {
                "backend": "gguf",
                "gguf_file": "m.gguf",
                "prompt_set": "recap_faithful",
                "n_ctx": 4096,
                "n_gpu_layers": 1,
                "generation": {"max_new_tokens": 64, "temperature": 0.0},
            }
        },
    }
