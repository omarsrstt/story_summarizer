"""A dead llama-server must say *why* it died.

The first real 467-chapter run ended at chapter 270 with nothing useful in the
traceback. The server's own log said ``CUDA error: out of memory``; the client
said "did not respond". Recovering from a crash is only useful if the crash is
named, so the log line is folded into the error.
"""

import pytest

from summarizer_gguf import GgufError, LlamaServer


def _server(tmp_path, log_body):
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"stub")
    log = tmp_path / "server.log"
    log.write_text(log_body, encoding="utf-8")
    return LlamaServer(binary, model, startup_timeout=5, log_path=log)


CUDA_OOM_LOG = """\
0.08.068.534 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 12288
35.32.082.594 W srv         alloc:  - making room for prompt cache entry, removing oldest entry (size = 349.080 MiB)
35.34.073.302 E CUDA error: out of memory
35.34.074.478 E   current device: 0, in function launch_mul_mat_q at mmq.cuh:1534
"""


def test_an_out_of_memory_death_reports_the_cuda_error(tmp_path):
    """The traceback must carry llama.cpp's own cause, not just 'did not respond'."""

    from test_summarizer_gguf import FakeProcess

    server = _server(tmp_path, CUDA_OOM_LOG)
    # The server really is gone, which is what a CUDA OOM leaves behind.
    server.process = FakeProcess(returncode=-1)

    report = server._stall_report(OSError("connection closed"))

    assert "out of memory" in report.lower(), report
    # Full accuracy means quoting llama.cpp, not the client's socket error.
    assert "server alive: no" in report or "cuda" in report.lower(), report


def test_a_cuda_error_is_recognised_as_a_crash(tmp_path):
    """``crashed()`` distinguishes 'llama.cpp died' from 'request was slow'."""

    server = _server(tmp_path, CUDA_OOM_LOG)
    assert server.crashed() is True

    healthy = _server(tmp_path, CUDA_OOM_LOG.split("35.34")[0])
    assert healthy.crashed() is False


def test_a_restart_does_not_erase_the_log_of_the_crash_it_recovered_from(tmp_path, monkeypatch):
    """The reason for a restart must survive the restart.

    ``log_path`` is keyed on the client's PID, so reopening it with "wb" on a
    reload truncated the log -- destroying the crash that prompted the reload,
    which was the only record of what went wrong.
    """

    import subprocess

    from test_summarizer_gguf import FakeProcess

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())

    server = _server(tmp_path, "0.08.068.534 I srv load_model: loading model\n")
    server._spawn(["stub"], {})
    server.stop()

    # The first child hit a fatal error before dying, as llama.cpp does.
    with open(server.log_path, "a", encoding="utf-8") as handle:
        handle.write("35.34.073.302 E CUDA error: out of memory\n")

    # Now the reload: the same server object spawns a second child.
    server._spawn(["stub"], {})
    try:
        kept = server.log_path.read_text(encoding="utf-8")
    finally:
        server.stop()

    assert "out of memory" in kept, "the crash log was truncated by the restart"
    assert "loading model" in kept, "the original startup log was lost"
