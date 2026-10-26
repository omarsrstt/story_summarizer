# Fast Chapter Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an HTTP-first scraper engine that scrapes a novel's chapters in the background with no visible window, keeping the existing Selenium engine as an explicit fallback.

**Architecture:** A one-off off-screen headful Chrome visit passes Cloudflare and harvests (a) the `cf_clearance` cookie + User-Agent and (b) the full chapter list from `select.select-chapter`. Both are cached. Every chapter is then a plain `curl_cffi` HTTP GET parsed with BeautifulSoup. Pure parsing/writing logic lives in dependency-light modules so it is unit-testable without a browser.

**Tech Stack:** Python 3.11, `uv`, `curl_cffi` (new, `scraper` extra), `beautifulsoup4`, `selenium` + `undetected-chromedriver`, `pytest` (new, dev), Windows.

**Spec:** `docs/superpowers/specs/2026-10-25-fast-scraper-design.md`

## Global Constraints

- Python 3.11; run everything via `uv run ...`. Paths resolve relative to the repo root (parent of `src/`).
- Modules import each other by **bare name** (`from scraper_parse import ...`) because scripts run with `src/` on `sys.path`. Configure pytest with `pythonpath = ["src"]`.
- `curl_cffi` is added **only** to the `scraper` optional extra; `pytest` only to a dev group.
- The browser must never be visible or steal focus: always `--window-position=-32000,-32000` (no `focus_window()`, no `maximize_window()`).
- Pin ChromeDriver with `version_main=` the installed Chrome major version (currently `155`).
- No fixed multi-second sleeps on the fast path.
- Existing chapters are **skipped by default**; `--overwrite` forces. Writes go to a temp file, renamed into place only after a plausibility check. A chapter is considered present if **either** `chapter_<n>.txt` (corpus form) **or** `chapter_<n:04d>.txt` (legacy form) exists.
- File name written is `chapter_<n>.txt` where `<n>` is the number parsed from the chapter **title** (not the URL index).
- Committed files must contain **no** real site names, URLs, or personal data. Only `src/scraper_config.example.json` (placeholders) is committed.

## Review Focus

Inputs/conditions the spec implies but no happy-path test covers; each gets a test in the owning task.

1. **HTTP 200 that is actually a challenge/partial page** — must never overwrite a good file (Task 2 plausibility + Task 3 temp-rename).
2. **Odd chapter titles** — `"c-1: Prologue: Start"`, `"Chapter 98"` (no subtitle), slugged recent titles — must map to the right number or be skipped, never mis-numbered (Task 1).
3. **Existing file present** (in either padded or unpadded form) — default must skip, not overwrite (Task 3).
4. **Cookie expiry mid-run under concurrency** — refresh once, not once per worker (Task 5).
5. **403 rate-limit vs 403 challenge** — must back off vs refresh, not conflate (Task 5).

---

### Task 1: Chapter-number parsing and map building

**Files:**
- Create: `src/scraper_parse.py`
- Create: `tests/test_scraper_parse.py`
- Modify: `pyproject.toml` (add `[tool.pytest.ini_options] pythonpath = ["src"]` and a dev group with `pytest`)

**Interfaces:**
- Produces:
  - `ChapterRef` — frozen dataclass `(number: int, slug: str, title: str)`
  - `parse_chapter_number(title: str) -> int | None`
  - `build_chapter_map(options: list[tuple[str, str]]) -> dict[int, ChapterRef]` (`options` are `(value, text)` pairs from `select.select-chapter`; `value` is the URL slug, `text` the title)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scraper_parse.py
from scraper_parse import ChapterRef, build_chapter_map, parse_chapter_number


def test_parse_number_from_titled_chapter():
    assert parse_chapter_number("Chapter 1: The Light At The End of The Tunnel") == 1

def test_parse_number_from_bare_chapter():
    assert parse_chapter_number("Chapter 98") == 98

def test_parse_number_ignores_prologue_and_junk():
    assert parse_chapter_number("c-1: Prologue: Start") is None
    assert parse_chapter_number("") is None

def test_build_map_uses_title_number_and_keeps_slug():
    options = [
        ("chapter-1", "c-1: Prologue: Start"),
        ("chapter-2", "Chapter 1: The Light At The End of The Tunnel"),
        ("chapter-530-end-story", "Chapter 530: End Story"),
    ]
    result = build_chapter_map(options)
    assert set(result) == {1, 530}
    assert result[1] == ChapterRef(1, "chapter-2", "Chapter 1: The Light At The End of The Tunnel")
    assert result[530].slug == "chapter-530-end-story"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scraper_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scraper_parse'`

- [ ] **Step 3: Implement**

`parse_chapter_number` uses `re.search(r"chapter\s+(\d+)", title, re.I)`; first match wins. `build_chapter_map` skips `None` numbers and de-dupes by number (first occurrence wins).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scraper_parse.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/scraper_parse.py tests/test_scraper_parse.py pyproject.toml
git commit -m "test(scraper): add pytest config and chapter-number parsing"
```

---

### Task 2: Content extraction and plausibility check

**Files:**
- Modify: `src/scraper_parse.py`
- Test: `tests/test_scraper_parse.py`

**Interfaces:**
- Produces:
  - `extract_chapter(html: str, content_selector: str, title_selector: str) -> tuple[str, str]` → `(title, text)`; `title` falls back to `""`, `text` to `""`.
  - `looks_like_chapter(text: str, min_chars: int = 400) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_scraper_parse.py
from scraper_parse import extract_chapter, looks_like_chapter

SAMPLE = """
<html><body>
  <div class="top"><span class="chapter">Chapter 4: My Life Now</span></div>
  <div class="txt"><p>Line one.</p><p>Line two.</p></div>
</body></html>
"""

def test_extract_returns_title_and_paragraph_text():
    title, text = extract_chapter(SAMPLE, "div.txt", "div.top span.chapter")
    assert title == "Chapter 4: My Life Now"
    assert "Line one." in text and "Line two." in text

def test_extract_missing_nodes_returns_empty():
    assert extract_chapter("<html></html>", "div.txt", "span.chapter") == ("", "")

def test_plausibility_rejects_short_or_challenge_text():
    assert looks_like_chapter("x" * 500) is True
    assert looks_like_chapter("Just a moment...") is False
    assert looks_like_chapter("") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scraper_parse.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_chapter'`

- [ ] **Step 3: Implement**

`extract_chapter` uses `BeautifulSoup(html, "html.parser")`, `select_one` for each selector, joins paragraph text with `"\n"`. `looks_like_chapter` returns `False` if text length `< min_chars` or it contains `"just a moment"` (case-insensitive).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scraper_parse.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/scraper_parse.py tests/test_scraper_parse.py
git commit -m "feat(scraper): extract chapter content and guard plausibility"
```

---

### Task 3: Chapter file writing (skip-by-default, atomic)

**Files:**
- Create: `src/scraper_store.py`
- Test: `tests/test_scraper_store.py`

**Interfaces:**
- Produces:
  - `chapter_filename(number: int) -> str` → `f"chapter_{number}.txt"`
  - `existing_chapter(save_root, novel_name: str, number: int) -> str | None` — returns the path of an existing `chapter_<n>.txt` or `chapter_<n:04d>.txt`, else `None`.
  - `write_chapter(save_root, novel_name: str, number: int, title: str, content: str, overwrite: bool = False) -> str | None` — returns path written, or `None` if skipped; writes `"{title}\n\n{content}"`, temp file then `os.replace`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scraper_store.py
from pathlib import Path
from scraper_store import chapter_filename, existing_chapter, write_chapter

def test_filename_is_unpadded():
    assert chapter_filename(7) == "chapter_7.txt"

def test_write_then_skip_without_overwrite(tmp_path):
    path = write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "body")
    assert path is not None and Path(path).read_text(encoding="utf-8").startswith("Chapter 7: X")
    assert write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "new") is None

def test_skip_when_legacy_padded_file_exists(tmp_path):
    d = tmp_path / "novel"
    d.mkdir()
    (d / "chapter_0007.txt").write_text("old", encoding="utf-8")
    assert existing_chapter(tmp_path, "novel", 7) is not None
    assert write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "new") is None

def test_overwrite_replaces_content(tmp_path):
    write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "old")
    write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "new", overwrite=True)
    assert (tmp_path / "novel" / "chapter_7.txt").read_text(encoding="utf-8").endswith("new")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scraper_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scraper_store'`

- [ ] **Step 3: Implement**

Create dirs; if `existing_chapter(...)` and `not overwrite`, return `None`; else write `<name>.tmp`, then `os.replace`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scraper_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/scraper_store.py tests/test_scraper_store.py
git commit -m "feat(scraper): atomic chapter writes that skip existing files"
```

---

### Task 4: Cookie/session cache

**Files:**
- Create: `src/scraper_session.py`
- Test: `tests/test_scraper_session.py`

**Interfaces:**
- Produces:
  - `SessionData` — dataclass `(user_agent: str, cookies: dict[str, str] = {}, chapter_map: list[tuple[str, str]] = [])`
  - `SessionCache(path)` with `.load() -> SessionData | None` and `.save(data: SessionData) -> None` (JSON; `None` on missing/corrupt file)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scraper_session.py
from scraper_session import SessionCache, SessionData

def test_roundtrip(tmp_path):
    cache = SessionCache(tmp_path / "c.json")
    assert cache.load() is None
    cache.save(SessionData("UA", {"cf_clearance": "v"}, [("chapter-2", "Chapter 1: X")]))
    loaded = cache.load()
    assert loaded.user_agent == "UA"
    assert loaded.cookies["cf_clearance"] == "v"
    assert loaded.chapter_map == [("chapter-2", "Chapter 1: X")]

def test_corrupt_file_loads_none(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{ not json", encoding="utf-8")
    assert SessionCache(p).load() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scraper_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scraper_session'`

- [ ] **Step 3: Implement**

Dataclass with defaults; `save` writes JSON; `load` wraps `json.loads` in `try/except -> None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scraper_session.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/scraper_session.py tests/test_scraper_session.py
git commit -m "feat(scraper): cache Cloudflare cookies and the chapter list"
```

---

### Task 5: HTTP fetch with challenge detection and single-flight refresh

**Files:**
- Modify: `src/scraper_session.py`
- Test: `tests/test_scraper_session.py`

**Interfaces:**
- Consumes: `SessionData`, `SessionCache`.
- Produces:
  - `ChallengeError(Exception)`
  - `FastSession(site: dict, novel_slug: str, cache: SessionCache, transport, harvest)` where `transport(url, headers, cookies) -> tuple[int, str]` and `harvest() -> SessionData`
  - `FastSession.fetch(url: str) -> str` — raises `ChallengeError` if the body is a challenge; on challenge calls `harvest()` **once**, re-saves the cache, retries the URL once. A refresh already in progress is not repeated (lock-guarded).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scraper_session.py (append)
import pytest
from scraper_session import ChallengeError, FastSession, SessionData

def _session(transport, harvest, tmp_path):
    return FastSession({"base_url": "https://x/"}, "novel", SessionCache(tmp_path / "c.json"), transport, harvest)

def test_fetch_returns_body(tmp_path):
    s = _session(lambda u, h, c: (200, "<html>ok</html>"), lambda: SessionData("UA", {}, []), tmp_path)
    assert s.fetch("https://x/book/novel/chapter-2") == "<html>ok</html>"

def test_fetch_refreshes_once_then_succeeds(tmp_path):
    calls = {"harvest": 0, "n": 0}
    def transport(u, h, c):
        calls["n"] += 1
        return (200, "Just a moment...") if calls["n"] == 1 else (200, "<html>real</html>")
    def harvest():
        calls["harvest"] += 1
        return SessionData("UA2", {"cf_clearance": "new"}, [])
    s = _session(transport, harvest, tmp_path)
    assert s.fetch("https://x/book/novel/chapter-2") == "<html>real</html>"
    assert calls["harvest"] == 1

def test_fetch_gives_up_after_one_refresh(tmp_path):
    s = _session(lambda u, h, c: (403, "Just a moment..."), lambda: SessionData("UA", {}, []), tmp_path)
    with pytest.raises(ChallengeError):
        s.fetch("https://x/book/novel/chapter-2")

def test_fetch_does_not_swallow_non_challenge_body(tmp_path):
    s = _session(lambda u, h, c: (404, "<html>not found</html>"), lambda: SessionData("UA", {}, []), tmp_path)
    assert "not found" in s.fetch("https://x/book/novel/chapter-2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scraper_session.py -v`
Expected: FAIL — `ImportError: cannot import name 'ChallengeError'`

- [ ] **Step 3: Implement**

`fetch` treats a body containing `"just a moment"` (case-insensitive) as a challenge regardless of status. On challenge it acquires a `threading.Lock`, harvests if no other thread already refreshed for this fetch attempt, saves the cache, and retries once; a second challenge raises `ChallengeError`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scraper_session.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/scraper_session.py tests/test_scraper_session.py
git commit -m "feat(scraper): HTTP fetch that refreshes cookies once on challenge"
```

---

### Task 6: Browser harvest of cookies + chapter list

**Files:**
- Modify: `src/scraper_session.py`
- Test: `tests/test_scraper_session.py`

**Interfaces:**
- Produces:
  - `harvest_session(site: dict, novel_slug: str, chrome_major: int, driver_factory) -> SessionData` — opens an off-screen driver, navigates to a chapter page, reads UA + cookies, clicks the chapter-list icon, reads `select.select-chapter` options as `(value, text)`, closes the driver.
  - `default_driver_factory(chrome_major: int)` — builds the off-screen `undetected_chromedriver` instance (no visible window).

`driver_factory` returns an object with `get(url)`, `execute_script(js)`, `find_elements(by, selector)`, `get_cookies()`, `quit()` (tests inject a stub).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scraper_session.py (append)
class _StubOption:
    def __init__(self, v, t): self._v, self.text = v, t
    def get_attribute(self, _): return self._v

class _StubSelect:
    def __init__(self, opts): self._opts = opts
    def find_elements(self, by, sel): return self._opts

class _StubDriver:
    def __init__(self): self.quit_called = False
    def get(self, url): pass
    def execute_script(self, js): return "UA"
    def get_cookies(self): return [{"name": "cf_clearance", "value": "abc"}]
    def find_elements(self, by, sel):
        if "select-chapter" in sel:
            return [_StubSelect([_StubOption("chapter-2", "Chapter 1: X")])]
        return [object()]
    def quit(self): self.quit_called = True

def test_harvest_reads_cookies_ua_and_options(tmp_path):
    driver = _StubDriver()
    site = {"base_url": "https://x/", "chapter_list_icon_selector": "i",
            "chapter_list_selector": "select.select-chapter"}
    data = harvest_session(site, "novel", 155, driver_factory=lambda major: driver)
    assert data.user_agent == "UA"
    assert data.cookies["cf_clearance"] == "abc"
    assert data.chapter_map == [("chapter-2", "Chapter 1: X")]
    assert driver.quit_called is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scraper_session.py -v`
Expected: FAIL — `ImportError: cannot import name 'harvest_session'`

- [ ] **Step 3: Implement**

UA via `execute_script("return navigator.userAgent")`. Click the icon via `execute_script` on the `chapter_list_icon_selector` element, then poll (bounded retry loop, not a 6.5 s sleep) until `select.select-chapter` has options. `default_driver_factory` sets `--window-position=-32000,-32000`, `--window-size=1280,900`, `version_main=chrome_major`, `headless=False`. Always `quit()` in a `finally`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scraper_session.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/scraper_session.py tests/test_scraper_session.py
git commit -m "feat(scraper): harvest cookies and chapter list off-screen"
```

---

### Task 7: Orchestration

**Files:**
- Create: `src/scraper_fast.py`
- Test: `tests/test_scraper_fast.py`

**Interfaces:**
- Consumes: `parse_chapter_input` (from `scraper`), `build_chapter_map`/`extract_chapter`/`looks_like_chapter` (from `scraper_parse`), `write_chapter` (from `scraper_store`), `FastSession`/`harvest_session`/`SessionCache`/`default_driver_factory` (from `scraper_session`).
- Produces:
  - `chapter_url(site: dict, novel_slug: str, slug: str) -> str`
  - `make_session(**kwargs) -> FastSession` — thin seam tests monkeypatch.
  - `run_fast(config: dict, site: dict, novel_slug: str, chapter_sel: str, save_dir: str, overwrite: bool, concurrency: int, refresh: bool, repo_root) -> int` — returns the number of chapters written.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scraper_fast.py
import scraper_fast as fast

class _FakeSession:
    def __init__(self, html): self.html = html
    def fetch(self, url): return self.html
    def chapter_map(self): return {1: fast.ChapterRef(1, "chapter-2", "Chapter 1: X")}

HTML = ('<div class="top"><span class="chapter">Chapter 1: X</span></div>'
        '<div class="txt">' + ("word " * 200) + "</div>")
SITE = {"base_url": "https://x/", "content_selector": "div.txt",
        "chapter_title_selector": "div.top span.chapter"}

def test_run_fast_writes_one_chapter(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "make_session", lambda **k: _FakeSession(HTML))
    written = fast.run_fast({"save_dir": "novels"}, SITE, "novel", "1", str(tmp_path), False, 2, False, tmp_path)
    assert written == 1
    assert (tmp_path / "novel" / "chapter_1.txt").exists()

def test_run_fast_skips_implausible_body(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "make_session", lambda **k: _FakeSession("<html>Just a moment...</html>"))
    assert fast.run_fast({"save_dir": "novels"}, SITE, "novel", "1", str(tmp_path), False, 2, False, tmp_path) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scraper_fast.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scraper_fast'`

- [ ] **Step 3: Implement**

`run_fast` builds the session via `make_session(...)`, gets `chapter_map()`, resolves the selection with `parse_chapter_input(chapter_sel, max(map))`, then for each number: fetch `chapter_url(...)`, `extract_chapter`, `looks_like_chapter` gate, `write_chapter`. Fetch with a bounded `concurrent.futures.ThreadPoolExecutor(concurrency)`; writes happen on the main thread as results arrive. Unmapped numbers are printed, not guessed. Re-export `ChapterRef` from `scraper_parse` so the test can reference `fast.ChapterRef`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scraper_fast.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/scraper_fast.py tests/test_scraper_fast.py
git commit -m "feat(scraper): fast engine orchestration"
```

---

### Task 8: CLI integration, dependencies, config, docs

**Files:**
- Modify: `src/scraper.py` (argparse + dispatch)
- Modify: `pyproject.toml` (add `curl_cffi` to the `scraper` extra)
- Modify: `src/scraper_config.example.json` (add `chapter_list_icon_selector`, `chapter_list_selector`)
- Modify: `README.md`
- Test: `tests/test_cli_engine.py`

**Interfaces:**
- Consumes: `run_fast`.
- Produces: `--engine {fast,legacy}` (default `fast`), `--concurrency N` (default `3`), `--overwrite`, `--refresh-cookies` on the existing parser.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_engine.py
import sys
import pytest
from scraper import parse_arguments

WEBSITES = [{"name": "Light Novel Pub"}]

def _parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["scraper.py", *argv])
    return parse_arguments(WEBSITES)

def test_engine_defaults_to_fast(monkeypatch):
    args = _parse(monkeypatch)
    assert args.engine == "fast"
    assert args.concurrency == 3
    assert args.overwrite is False

def test_engine_legacy_selectable(monkeypatch):
    assert _parse(monkeypatch, "--engine", "legacy").engine == "legacy"

def test_concurrency_and_overwrite(monkeypatch):
    args = _parse(monkeypatch, "--concurrency", "5", "--overwrite")
    assert args.concurrency == 5 and args.overwrite is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_engine.py -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'engine'`

- [ ] **Step 3: Implement**

Add the four arguments in `parse_arguments`. Restructure `main()` so args are parsed **before** any driver is created: after `get_missing_args`, if `args.engine == "fast"` then `from scraper_fast import run_fast` (lazy, to avoid the circular import) and `return run_fast(...)`; only the legacy branch calls `setup_driver()`. Add `chapter_list_icon_selector`/`chapter_list_selector` to both site entries in `scraper_config.example.json` and to the local `scraper_config.json`. Add `curl_cffi` to the `scraper` extra in `pyproject.toml`; run `uv sync --extra scraper`. Update the README: the fast engine is the default, `--engine legacy` is the fallback, and describe the one-off off-screen browser + cached cookies; keep it free of real site names/URLs.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest -v`
Expected: PASS (all tests)

- [ ] **Step 5: Manual end-to-end check**

Run: `uv run python src/scraper.py --engine fast --novel the-beginning-after-the-end --chapter_num 1 --concurrency 3`
Expected: no window appears; a `cf_clearance` harvest happens once (off-screen); `novels/the-beginning-after-the-end/chapter_1.txt` is skipped (already exists); exit code 0.

- [ ] **Step 6: Commit**

```bash
git add src/scraper.py pyproject.toml src/scraper_config.example.json README.md tests/test_cli_engine.py uv.lock
git commit -m "feat(scraper): wire the fast engine in as the default CLI"
```

---

## Verification (whole feature)

- `uv run pytest -v` is green.
- Fast-fetched chapter 1 paragraph text equals a fresh legacy scrape of chapter 1.
- Requesting a number absent from the site's chapter list (e.g. 510, since the
  list jumps 509→513) is reported and skipped; no file is written.
- A 20-chapter run with `--concurrency 4` completes with no visible window and never takes focus.
- Blanking the cached cookie triggers exactly one off-screen refresh, then the run continues.
- `git grep -i` over `HEAD` for real site names/URLs finds nothing outside the gitignored real config.
