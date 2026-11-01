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
import re
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

#: Loopback only: this server must never be reachable from the network.
HOST = "127.0.0.1"

#: llama.cpp prints no stable ready-marker on stdout, so readiness is polled.
POLL_INTERVAL = 0.25

#: An unbounded server log grows all run (~900 B per request), and a reload
#: appends to it rather than starting over. Past this size the oldest half is
#: dropped, keeping the recent crash evidence and the file small.
LOG_MAX_BYTES = 8 * 1024 * 1024


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
        log_path=None,
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
        #: Where llama-server's output is written. It cannot be a pipe: nobody
        #: reads it during a run, so the 4 KiB OS buffer silently swallows the
        #: startup banner and any load error (see ``_spawn``).
        self.log_path = Path(log_path) if log_path else (
            Path(tempfile.gettempdir()) / f"llama-server-{os.getpid()}.log"
        )
        self._log_handle = None

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

        self.process = self._spawn(cmd, env)

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

    def _spawn(self, cmd: list[str], env: dict) -> subprocess.Popen:
        """Start llama-server with its output going to a log file.

        The previous version captured the child's streams into ``subprocess.PIPE``
        and never read them. That is how the *diagnostics* were lost: the OS pipe
        buffer is only 4 KiB on Windows, and llama.cpp writes far more than that
        during startup, so everything past the first few kB was silently dropped.
        ``_exit_reason`` then reported an empty cause for a model that failed to
        load.

        Measured behaviour: a full pipe does not stall this build (it drops the
        output), so this was not a deadlock. It is still not something to rely
        on -- pipe blocking is a real hazard on other platforms and other log
        volumes, and a file has no such limit. The file also keeps the log around
        for the post-mortem that ``_stall_report`` and ``_exit_reason`` need.

        A restart appends rather than truncates. ``log_path`` is keyed on the
        client's PID, so it is the same file before and after a reload, and
        reopening it with "wb" used to erase the crash that prompted the reload.
        """

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the file bounded: a 467-chapter run writes ~900 B per request,
        # so an unbounded append across several reloads grows without limit.
        # Trimming only when already large keeps the recent crash evidence.
        if self.log_path.is_file() and self.log_path.stat().st_size > LOG_MAX_BYTES:
            self._trim_log()
        self._log_handle = open(self.log_path, "ab")
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

    def _trim_log(self) -> None:
        """Cut an oversized log back to its most recent half.

        Best effort: a log that cannot be trimmed is still a valid log, and
        losing the run over housekeeping would be worse than a huge file.
        """

        try:
            keep = LOG_MAX_BYTES // 2
            data = self.log_path.read_bytes()[-keep:]
            self.log_path.write_bytes(data)
        except OSError:
            pass

    def stop(self) -> None:
        """Terminate the server, releasing its VRAM. Safe to call twice."""

        process, self.process = self.process, None
        self.port = None

        handle, self._log_handle = self._log_handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:  # pragma: no cover - already closed
                pass

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
        """Explain a startup death using the tail of llama-server's own log."""

        tail = "\n".join(self._log_tail().strip().splitlines()[-8:])
        return (
            f"llama-server exited during startup for {self.model.name}"
            + (f":\n{tail}" if tail else "")
        )

    def alive(self) -> bool:
        """Whether the child process is still running.

        Used to tell "slow" from "dead" when a request stops responding; the
        HTTP client cannot distinguish the two on its own.
        """

        return self.process is not None and self.process.poll() is None

    def _settled_alive(self, timeout: float = 2.0) -> bool:
        """``alive()`` re-checked until the child's state is definite.

        A killed child keeps reporting alive for ~200ms before Windows signals
        the process handle, so a single ``poll()`` straight after a failed
        request says "alive" for a server that has already died. Polling until
        the answer is stable costs at most ``timeout`` seconds on a real stall,
        and only on the failure path.
        """

        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.alive():
                return False
            time.sleep(0.1)
        return self.alive()

    def vram_report(self) -> str:
        """A one-line GPU memory reading, or "" when nvidia-smi is unavailable.

        An 8 GB card is the binding constraint on this project, so a stall is
        far more informative with the memory figure beside it than without.
        """

        try:
            done = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        line = done.stdout.strip().splitlines()
        return f"VRAM: {line[0].strip()}" if line else ""

    def _stall_report(self, exc: Exception) -> str:
        """Explain a request that got no answer, using what the server left behind.

        A bare "did not respond" cannot separate a GPU eviction from a wedged
        lock. The last log line, the process state and the VRAM figure usually
        can, and they cost nothing to collect once the request has already
        failed.

        A server that died is reported as dead and quotes llama.cpp's own last
        error. An OOM on the 8 GB card looks like a hang from the client's side,
        and "did not respond" sent the first investigation after a client-side
        cause that was never there.
        """

        dead = not self._settled_alive()
        lines = [f"llama-server did not respond for {self.model.name}: {exc}"]
        lines.append(f"server alive: {'no' if dead else 'yes'}")

        vram = self.vram_report()
        if vram:
            lines.append(vram)

        if dead:
            culprit = self._last_error()
            if culprit:
                lines.append(f"llama.cpp said: {culprit}")

        tail = [line for line in self._log_tail().splitlines() if line.strip()]
        if tail:
            lines.append(f"last log line: {tail[-1].strip()}")
            lines.append(f"log: {self.log_path}")
        return "\n".join(lines)

    def crashed(self) -> bool:
        """Whether llama.cpp hit an error that killed it, rather than stalling.

        Only called after a request has already failed, so reading the log costs
        nothing. Any ``E`` line is llama.cpp flagging a fatal condition; an
        ordinary slow request never writes one.
        """

        return self._last_error() is not None

    def _last_error(self) -> str | None:
        """llama.cpp's most recent fatal block, or None if it logged no error.

        The cause and its context straddle several lines -- the real failure was
        ``CUDA error: out of memory`` followed by ``in function launch_mul_mat_q
        at mmq.cuh:1534``. Reporting only the last line loses which call failed,
        so the whole trailing block is returned with timestamps stripped.
        """

        lines = self._log_tail().splitlines()
        block: list[str] = []
        for line in reversed(lines):
            if re.match(r"^\d+\.\d+\.\d+\.\d+ E\s", line):
                block.append(re.sub(r"^\d+\.\d+\.\d+\.\d+ E\s+", "", line).strip())
            elif block:
                break
        if not block:
            return None
        return " / ".join(reversed(block))

    def _log_tail(self, limit: int = 65536) -> str:
        """The last ``limit`` bytes of the child's log, best effort."""

        try:
            with open(self.log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - limit))
                return handle.read().decode("utf-8", "replace")
        except OSError:
            return ""

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
            raise GgufError(self._stall_report(exc)) from exc

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
