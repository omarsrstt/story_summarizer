"""Tests for the llama.cpp (GGUF) HTTP backend.

The backend runs ``llama-server`` as a child process instead of loading weights
into this Python process. Two reasons:

* the model is loaded **once** and then answers many requests, so a 516-chapter
  run does not pay a reload per chapter;
* killing the child destroys its CUDA context, which is what actually returns
  VRAM to the system -- a model resident in this process cannot do that.

Requests go to llama-server's OpenAI-compatible endpoint so each model's chat
template is applied from the GGUF metadata, matching what ``apply_chat_template``
does on the transformers path.
"""

import json
import subprocess
import urllib.error

import pytest

from summarizer_gguf import (
    GgufError,
    LlamaServer,
    build_server_command,
    chat_payload,
    extract_message,
    find_binary,
    _free_port,
)


class FakeProcess:
    """Stand-in for subprocess.Popen that records lifecycle calls."""

    def __init__(self, returncode=None):
        self.terminated = False
        self.killed = False
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = json.dumps(body).encode("utf-8") if not isinstance(body, bytes) else body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------
# command construction
# --------------------------------------------------------------------------

def test_command_points_at_the_model_and_binary(tmp_path):
    binary = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    cmd = build_server_command(binary, model, port=8123, n_ctx=16384, n_gpu_layers=999)

    assert cmd[0] == str(binary)
    assert cmd[cmd.index("-m") + 1] == str(model)
    assert cmd[cmd.index("--port") + 1] == "8123"
    assert cmd[cmd.index("-c") + 1] == "16384"
    assert cmd[cmd.index("-ngl") + 1] == "999"


def test_command_binds_to_loopback_only(tmp_path):
    """The server must not be reachable from the network."""
    cmd = build_server_command(
        tmp_path / "b.exe", tmp_path / "m.gguf", port=1, n_ctx=8, n_gpu_layers=1
    )
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_command_enables_jinja_chat_templates(tmp_path):
    """Without jinja the model's chat template is skipped and quality drops."""
    cmd = build_server_command(
        tmp_path / "b.exe", tmp_path / "m.gguf", port=1, n_ctx=8, n_gpu_layers=1
    )
    assert "--jinja" in cmd
    assert "--no-webui" in cmd


def test_command_disables_reasoning_by_default(tmp_path):
    """Thinking traces must not leak into a recap."""
    cmd = build_server_command(
        tmp_path / "b.exe", tmp_path / "m.gguf", port=1, n_ctx=8, n_gpu_layers=1
    )
    assert "--reasoning" in cmd
    assert cmd[cmd.index("--reasoning") + 1] == "off"


def test_command_can_leave_reasoning_on(tmp_path):
    cmd = build_server_command(
        tmp_path / "b.exe", tmp_path / "m.gguf", port=1, n_ctx=8,
        n_gpu_layers=1, reasoning=True,
    )
    assert cmd[cmd.index("--reasoning") + 1] == "on"


# --------------------------------------------------------------------------
# request payload
# --------------------------------------------------------------------------

def test_payload_carries_messages_and_greedy_settings():
    messages = [
        {"role": "system", "content": "Recap faithfully."},
        {"role": "user", "content": "chapter text"},
    ]
    payload = chat_payload(messages, max_new_tokens=700, temperature=0.0)

    assert payload["messages"] == messages
    assert payload["max_tokens"] == 700
    assert payload["temperature"] == 0.0
    assert payload["stream"] is False


def test_payload_is_json_serialisable():
    payload = chat_payload([{"role": "user", "content": "x"}], max_new_tokens=8)
    assert json.loads(json.dumps(payload))["max_tokens"] == 8


# --------------------------------------------------------------------------
# response handling
# --------------------------------------------------------------------------

def test_extract_message_returns_content():
    body = {"choices": [{"message": {"role": "assistant", "content": "  A recap.  "}}]}
    assert extract_message(body) == "A recap."


def test_extract_message_never_returns_reasoning_traces():
    body = {
        "choices": [
            {
                "message": {
                    "content": "A recap.",
                    "reasoning_content": "Hmm, let me think about chapter 3...",
                }
            }
        ]
    }
    out = extract_message(body)
    assert out == "A recap."
    assert "let me think" not in out


def test_extract_message_raises_when_there_is_no_choice():
    with pytest.raises(GgufError):
        extract_message({"choices": []})


def test_extract_message_raises_on_a_present_but_empty_choice():
    with pytest.raises(GgufError):
        extract_message({"choices": [{"message": {"content": ""}}]})


# --------------------------------------------------------------------------
# binary discovery
# --------------------------------------------------------------------------

def test_find_binary_accepts_an_explicit_path(tmp_path):
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"stub")
    assert find_binary(str(exe)) == exe


def test_find_binary_explains_how_to_get_one(tmp_path):
    with pytest.raises(GgufError) as err:
        find_binary(str(tmp_path / "nope.exe"))
    message = str(err.value)
    assert "llama-server" in message
    assert "releases" in message


# --------------------------------------------------------------------------
# server lifecycle
# --------------------------------------------------------------------------

def test_free_port_returns_an_unused_port():
    a = _free_port()
    b = _free_port()
    assert isinstance(a, int) and 1024 < a < 65536
    assert a != b


def test_server_reports_a_missing_model(tmp_path):
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"stub")
    with pytest.raises(GgufError) as err:
        LlamaServer(binary, tmp_path / "absent.gguf")
    assert "not found" in str(err.value).lower()


def test_start_waits_for_health_then_reports_ready(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"stub")

    proc = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)

    calls = {"n": 0}

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("not up yet")
        return FakeResponse({"status": "ok"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    server = LlamaServer(binary, model, startup_timeout=10)
    server.start()
    assert calls["n"] == 3
    server.stop()
    assert proc.terminated or proc.killed


def test_start_raises_if_the_server_dies_while_booting(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"stub")

    proc = FakeProcess(returncode=1)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )

    server = LlamaServer(binary, model, startup_timeout=2)
    with pytest.raises(GgufError):
        server.start()


def test_start_raises_when_it_never_becomes_healthy(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"stub")

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )

    server = LlamaServer(binary, model, startup_timeout=0.3)
    with pytest.raises(GgufError) as err:
        server.start()
    assert "start" in str(err.value).lower()


def test_chat_requires_start(tmp_path):
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"stub")
    server = LlamaServer(binary, model)
    with pytest.raises(GgufError) as err:
        server.chat([{"role": "user", "content": "hi"}])
    assert "not running" in str(err.value).lower()


def test_chat_posts_to_the_completions_endpoint(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"stub")

    seen = {}

    class Req:
        def __init__(self, url, data=None, headers=None, method=None):
            self.full_url = url
            self.data = data
            self.headers = headers or {}

        def get_header(self, name, default=None):
            return self.headers.get(name, default)

    def fake_urlopen(url_or_req, timeout=None, data=None):
        # health check
        if not hasattr(url_or_req, "data"):
            return FakeResponse({"status": "ok"})
        seen["url"] = url_or_req.full_url
        seen["body"] = json.loads(url_or_req.data.decode("utf-8"))
        return FakeResponse(
            {"choices": [{"message": {"content": "The recap."}}]}
        )

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("urllib.request.Request", Req)

    server = LlamaServer(binary, model, startup_timeout=5)
    server.start()
    out = server.chat(
        [
            {"role": "system", "content": "Recap."},
            {"role": "user", "content": "text"},
        ],
        max_new_tokens=128,
    )

    assert out == "The recap."
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["max_tokens"] == 128
    assert seen["body"]["messages"][0]["role"] == "system"


def test_chat_surfaces_an_http_error_with_the_body(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"stub")

    class Req:
        def __init__(self, url, data=None, headers=None, method=None):
            self.full_url = url
            self.data = data
            self.headers = headers or {}

        def get_header(self, name, default=None):
            return self.headers.get(name, default)

    def fake_urlopen(url_or_req, timeout=None, data=None):
        if not hasattr(url_or_req, "data"):
            return FakeResponse({"status": "ok"})
        raise urllib.error.HTTPError(
            "http://x", 400, "Bad Request", {}, FakeResponse(b'{"error":"context too long"}')
        )

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("urllib.request.Request", Req)

    server = LlamaServer(binary, model, startup_timeout=5)
    server.start()
    with pytest.raises(GgufError) as err:
        server.chat([{"role": "user", "content": "x"}])
    assert "context too long" in str(err.value)


def test_stop_is_idempotent(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"stub")

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: FakeResponse({}))

    server = LlamaServer(binary, model, startup_timeout=5)
    server.start()
    server.stop()
    server.stop()  # must not raise


def test_context_manager_releases_the_server(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"stub")

    proc = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: FakeResponse({}))

    with LlamaServer(binary, model, startup_timeout=5):
        pass
    assert proc.terminated or proc.killed
