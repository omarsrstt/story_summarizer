"""llama.cpp (GGUF) backend for the summarizer.

``summarizer.py`` runs a Hugging Face model *inside* this Python process, with
bitsandbytes doing 4-bit quantization at load time. That works, but it has two
drawbacks once a run is measured in hours:

* bnb 4-bit inference is several times slower than llama.cpp's kernels on the
  same card, and
* weights loaded here are only released when the process dies, so "pause and
  give the GPU back" is not really possible.

So this module drives the official ``llama-server`` binary as a **child
process** instead. The model is loaded once and answers many requests, and
killing the child tears down its CUDA context, returning the VRAM to the
system. That is what makes a genuine pause/resume possible.

Requests go through llama-server's OpenAI-compatible endpoint, so each model's
chat template is applied from the GGUF metadata -- the same effect as
``tokenizer.apply_chat_template()`` on the transformers path. Reasoning traces
are switched off by default so thinking models cannot leak deliberation into a
chapter recap.

Nothing here is imported unless a model's config entry names a ``gguf`` file,
so the existing transformers path is untouched.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

#: Loopback only: this server must never be reachable from the network.
HOST = "127.0.0.1"

#: llama.cpp prints no stable ready-marker on stdout, so readiness is polled.
POLL_INTERVAL = 0.25


class GgufError(RuntimeError):
    """A GGUF model could not be run."""


def find_binary(binary_path: str | None) -> Path:
    """Return the llama-server executable, or explain how to obtain one."""

    if not binary_path:
        raise GgufError(
            "No llama-server binary configured. Set 'llama_binary' in the model's "
            "config entry, or put llama-server on PATH. Prebuilt Windows CUDA "
            "builds: https://github.com/ggml-org/llama.cpp/releases"
        )
    path = Path(binary_path)
    if not path.exists():
        raise GgufError(
            f"llama-server not found at {path}. Download a prebuilt build from "
            "https://github.com/ggml-org/llama.cpp/releases and point "
            "'llama_binary' at llama-server.exe."
        )
    return path


def _free_port() -> int:
    """Ask the OS for an unused loopback port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def build_server_command(
    binary: Path,
    model: Path,
    *,
    port: int,
    n_ctx: int,
    n_gpu_layers: int,
    reasoning: bool = False,
    threads: int | None = None,
) -> list[str]:
    """Assemble the llama-server invocation.

    ``--jinja`` is what makes ``/v1/chat/completions`` apply the model's own chat
    template; without it the system prompt is delivered as plain text and recap
    quality drops. ``--no-webui`` keeps it a pure API, and binding to loopback
    keeps it off the network.
    """

    cmd = [
        str(binary),
        "-m", str(model),
        "--host", HOST,
        "--port", str(port),
        "-c", str(n_ctx),
        "-ngl", str(n_gpu_layers),
        "--jinja",
        "--no-webui",
        "--reasoning", "on" if reasoning else "off",
    ]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


def chat_payload(
    messages: list[dict],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    no_repeat_ngram_size: int | None = None,
) -> dict:
    """Build an OpenAI-style chat completion body."""

    payload: dict = {
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
        # llama.cpp samples greedily at temperature 0, matching the transformers
        # path's do_sample=False.
        "cache_prompt": False,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if no_repeat_ngram_size:
        # Mirrors the transformers generation setting; llama.cpp spells it
        # differently and silently ignores unknown keys, so guard it.
        payload["repeat_penalty"] = 1.05
    return payload


def extract_message(body: dict) -> str:
    """Return the assistant's content, never its reasoning trace."""

    choices = body.get("choices") or []
    if not choices:
        raise GgufError(f"llama-server returned no choices: {str(body)[:200]}")

    message = choices[0].get("message") or {}
    content = message.get("content") or ""

    # Some templates put everything in ``reasoning_content`` when the thought
    # tags are not stripped. A recap must come from ``content`` only.
    if isinstance(content, str):
        content = content.strip()
    if not content:
        raise GgufError(
            "llama-server returned an empty completion. The model may have spent "
            "its token budget on reasoning; set 'reasoning': false in the config."
        )
    return content


class LlamaServer:
    """A llama-server child process answering chat completion requests."""

    def __init__(
        self,
        binary,
        model,
        *,
        n_ctx: int = 16384,
        n_gpu_layers: int = 999,
        temperature: float = 0.0,
        reasoning: bool = False,
        threads: int | None = None,
        startup_timeout: float = 600.0,
        request_timeout: float = 900.0,
    ):
        self.binary = find_binary(str(binary)) if binary else find_binary(None)
        self.model = Path(model)
        if not self.model.exists():
            raise GgufError(f"GGUF model not found: {self.model}")

        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.reasoning = reasoning
        self.threads = threads
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout

        self.port: int | None = None
        self.process: subprocess.Popen | None = None
        #: Usage/timing counters from the most recent request, for benchmarking.
        self.last_stats: dict = {}
        #: Why the last generation stopped: "stop", "length", or None.
        self.last_finish_reason: str | None = None

    # -- lifecycle ------------------------------------------------------

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise GgufError("llama-server is not running")
        return f"http://{HOST}:{self.port}"

    def start(self) -> "LlamaServer":
        """Launch the server and wait until it answers health checks."""

        if self.process is not None and self.process.poll() is None:
            return self

        self.port = _free_port()
        cmd = build_server_command(
            self.binary,
            self.model,
            port=self.port,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            reasoning=self.reasoning,
            threads=self.threads,
        )

        env = dict(os.environ)
        # llama.cpp's CUDA build needs its own DLLs beside the executable.
        env["PATH"] = f"{self.binary.parent}{os.pathsep}{env.get('PATH', '')}"

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise GgufError(self._exit_reason())
            if self._healthy():
                return self
            time.sleep(POLL_INTERVAL)

        self.stop()
        raise GgufError(
            f"llama-server did not start within {self.startup_timeout:.0f}s "
            f"for {self.model.name}"
        )

    def stop(self) -> None:
        """Terminate the server, releasing its VRAM. Safe to call twice."""

        process, self.process = self.process, None
        self.port = None
        if process is None or process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _exit_reason(self) -> str:
        stderr = ""
        stream = getattr(self.process, "stderr", None)
        if stream is not None:
            try:
                stderr = stream.read().decode("utf-8", "replace")
            except Exception:  # pragma: no cover - stderr already consumed
                stderr = ""
        tail = "\n".join(stderr.strip().splitlines()[-8:])
        return (
            f"llama-server exited during startup for {self.model.name}"
            + (f":\n{tail}" if tail else "")
        )

    # -- requests -------------------------------------------------------

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 700,
        temperature: float | None = None,
    ) -> str:
        """Send one chat completion request and return the recap text."""

        if self.process is None or self.port is None or self.process.poll() is not None:
            raise GgufError("llama-server is not running; call start() first")

        payload = chat_payload(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature if temperature is None else temperature,
        )
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:  # pragma: no cover - body already consumed
                detail = ""
            raise GgufError(
                f"llama-server rejected the request ({exc.code}): {detail[:300]}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise GgufError(
                f"llama-server did not respond for {self.model.name}: {exc}"
            ) from exc

        self.last_stats = body.get("usage") or {}
        try:
            choice = body["choices"][0]
        except (KeyError, IndexError):
            choice = {}
        # "length" means llama.cpp stopped at the token cap mid-sentence, which
        # is worth surfacing: a truncated summary looks complete but is not.
        self.last_finish_reason = choice.get("finish_reason")
        return extract_message(body)

    def tokenize(self, text: str) -> int:
        """Return the server's own token count for ``text``.

        Counted by the model's real tokenizer rather than estimated from
        characters, so the context budget is enforced exactly. A chapter that
        exceeds ``n_ctx`` would otherwise be truncated by llama.cpp and
        summarised from its opening only, with no error raised.
        """

        if self.process is None or self.port is None or self.process.poll() is not None:
            raise GgufError("llama-server is not running; call start() first")

        request = urllib.request.Request(
            f"{self.base_url}/tokenize",
            data=json.dumps({"content": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise GgufError(
                f"llama-server could not tokenize ({exc.code}); "
                "the build may predate the /tokenize endpoint"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise GgufError(f"llama-server did not respond while tokenizing: {exc}") from exc

        tokens = body.get("tokens")
        if not isinstance(tokens, list):
            raise GgufError(f"unexpected /tokenize response: {str(body)[:200]}")
        return len(tokens)
