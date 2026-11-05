# AGENTS.md

Personal novel scraper + local-LLM summarizer. Two-stage pipeline, both CLI scripts in `src/`.
A parallel comic path (`comic_scraper.py` → `comic_reader.py` → the same `summarize.py`) is a third
stage described below.

## Environment

- Dependencies are managed with **uv**. `pyproject.toml` is the source of truth and `uv.lock`
  pins exact versions; Python 3.11 is pinned in `.python-version`. Create the env with
  `uv sync`.
- Only the summarizer's core deps (torch, transformers, accelerate, bitsandbytes, tqdm) are in
  the default install. Scraper and web deps are optional extras:
  `uv sync --extra scraper --extra web`.
- `torch` is deliberately pinned to PyTorch's **cu130** wheel index via `[tool.uv.sources]` —
  the plain PyPI wheel for Windows is CPU-only, so it must not be used.
- Run scripts through uv so they use the project env, e.g.
  `uv run python src/summarizer.py`, `uv run python src/scraper.py` (they locate configs
  relative to `__file__`).

## Pipeline

1. `src/scraper.py` — Selenium + `undetected_chromedriver` (headful Chrome required) scrapes
   chapters into `novels/<slug>/chapter_NNNN.txt`. Site CSS selectors live in
   `src/scraper_config.json`, but note some selectors/navigation are still hardcoded in the
   script (e.g. `ul-list5`, `glyphicon-list-alt`, `select-chapter`) — config does NOT cover all
   of it.
2. `src/summarizer.py` — loads a Hugging Face model locally and writes summaries to
   `summaries/<slug>/`. Model behavior (chunk size, quantization, generation params, prompt
   template) is driven by `src/summarizer_config.json`.
3. `src/summarize.py` — **the summarizer to use.** Same config and output tree, but drives
   `llama-server` (llama.cpp) with a GGUF model instead of loading weights in-process, which is
   ~6x faster on this card and lets the model be unloaded to free VRAM. One recap per chapter at
   `summaries/<slug>/summary_chapter_NNNNN.txt`. Resumes by default (existing recaps are skipped,
   `--overwrite` to regenerate) and refuses to summarise a chapter whose prompt would overflow
   `n_ctx` rather than silently recapping only its opening.

   The same script builds three wider artifacts from those recaps (never from the source, so any
   of them can be rebuilt at a different width without re-reading the novel):

   | Flag | Output | What it is |
   | --- | --- | --- |
   | `-g N` / `--group-size N` | `summary_chapters_00001-00005.txt` | one summary per N chapters |
   | `--digest` | `digest.txt` | `Chapter 7: event. event.` — one or two sentences per chapter |
   | `--all` / `--master` | `master_summary.txt` | whole-novel summary, compressed in rounds |

   Flags combine (`-g 5 --digest --all`) and run in dependency order after the recaps they read.
   A missing recap is a hard error for these steps, not something silently skipped, because a
   batch or digest with a hole still *looks* complete.

   `-g N` and `--all` are the same merge at different scales; only the context window
   distinguishes them. `--all` measures real token counts each round and merges as many pieces as
   fit (`master.fan_in` per step), repeating until one piece remains. If the final merge hits the
   token cap it says so — a truncated summary otherwise reads as finished.

Prompts live in one shared `prompts` block in the config, referenced per model by `prompt_set`,
so every model is measured against identical wording. GGUF entries add `backend`, `gguf_file`
(looked up in `MODEL_DIR`), `n_ctx`, `n_gpu_layers` and `reasoning`. The default model is
`qwen3.5-9b`, chosen by a judged comparison of four 2026 GGUFs on 15 chapters.

The wider artifacts each use their own prompt set from that same block: `recap_faithful` for
chapters, `merge` for `--group-size`, `digest` for `--digest`, and `master` for `--all`. Sizing
lives in top-level `digest` (`batch_size`, `max_new_tokens`, `digest_file`) and `master`
(`fan_in`, `target_words`) blocks. `digest.batch_size` is only an upper bound — batches are also
split by measured token count, so a run of long recaps cannot overflow `n_ctx`.

Paths in config (`novels`, `summaries`, `NOVEL_DIR`, `save_dir`, `MODEL_DIR`, `LLAMA_BINARY`)
are resolved relative to the **repo root** (parent of `src/`), not the current working directory.
`tools/` (the ~1.3 GB prebuilt llama.cpp runtime) is gitignored and must be downloaded, not
committed.

`llama-server`'s own output goes to `%TEMP%\llama-server-<pid>.log`, not to the terminal and not
to a pipe. That file is the first place to look when a run dies or stalls: it holds the load
banner, per-request `slot print_timing` lines, and any load error. Capturing it in a pipe and
never reading it silently discarded everything past 4 KiB, which is why a failed model load used
to report an empty cause. A request that gets no answer now raises with `server alive:`,
a `VRAM:` reading, llama.cpp's own `Error` block when it logged one, and the last log line
attached, so a dead server can be told from a slow one.

## VRAM budget and the CUDA OOM (measured)

At `n_ctx=12288` the `qwen3.5-9b` model occupies **6694 MiB of the card's 8192 MiB** once loaded
— 1498 MiB of headroom, which is all the prompt cache and compute buffers have to live in.

A real 467-chapter run died at chapter 270 with:

```
alloc: - making room for prompt cache entry, removing oldest entry (size = 349.080 MiB)
E CUDA error: out of memory
E   current device: 0, in function launch_mul_mat_q at .../mmq.cuh:1534
```

So the failure is a genuine out-of-memory during a long run, not a hang and not a deadlock. A
CUDA OOM tears down the whole process, which is why the client only saw a closed socket. If it
recurs, lower `n_ctx` for the model entry (`8192` roughly halves the KV cache) or reduce
`n_gpu_layers` to spill some weights to system RAM. Note that `-np`/`--parallel` is *not* a
lever here: with `kv_unified` (this build's default) the KV buffer is one shared `-c`-sized pool
regardless of slot count.

A dead server is now restarted automatically, and if it cannot be restarted the run **aborts**
with the cause instead of labelling every remaining chapter failed. That cascade — one crash at
chapter 270 reported as 244 failed chapters — is what made the first diagnosis so hard.

The OOM is **intermittent, not tied to a particular chapter**, and it is survivable. The
completion run (441–513) hit it again after ~8.8 min at chapter 473, logged once, reloaded in
~9 s and carried on; 72 of 73 chapters completed and the straggler succeeded on a fresh server.
So the practical remedy is the restart path plus a re-run, not a config change: a lower `n_ctx`
would make the model refuse chapters whose prompt no longer fits, which is a worse trade than
occasionally re-summarising one chapter.

That log is also why `_spawn` **appends**: `log_path` is keyed on the client's PID and is reused
across reloads, so opening it `"wb"` erased the crash that prompted the reload. It now appends
(past `LOG_MAX_BYTES` it keeps only the recent half), so a restarted run still contains both the
error and the reload. The "chapter N FAILED" line is expected and re-runnable in that case — it
means the reload worked, not that the chapter is unsummarisable.

## Comic pipeline (separate from the novel one)

The comic is an **adaptation** of `novels/the-beginning-after-the-end/`, not a 1:1 transcription —
same story, different pacing and occasionally different scenes. Never index-join the two trees.
Its value is as *ground truth for OCR*: when a comic transcript is garbled, the novel text is what
tells you what the panel probably said.

Three scripts, each writing the next stage's input:

1. `src/comic_scraper.py` — downloads page images to `comics/<name>/c_<ordinal>_<page>.<ext>`.
   Also writes `comics/<name>/chapters.json`, the **authoritative manifest**.
2. `src/comic_reader.py` — OCRs the pages into `novels/comic-<name>/chapter_NNNNN.txt`, one file
   per chapter, using the same `chapter_%05d` convention the novel scraper uses.
3. `uv run python src/summarize.py -n comic-<name>` — unchanged from the novel path; it just sees
   another slug.

So the join between comic and novel stages is **plain integers on disk**. Neither `comic_reader.py`
nor `summarize.py` knows the comic scraper exists, and neither was modified for it. That is why the
manifest pins ordinals rather than trusting site numbers.

**Ordinal = reading-order position, not the site's chapter number.** They differ constantly. The
CDN files chapter N under `chapter_<N-1>`, and a re-upload can publish the same number twice
(`Chapter 144` and `Chapter 144.0` are *different chapters*, both numbered 144.0). The site also
inserts chapters retroactively — all of `175.1`–`175.19` appeared between `175` and `176` — and
lists `175.1` *before* `175`. So a range is resolved by position in the list; when a pinned
manifest has drifted and a range would walk past a chapter the site places inside it, the run
warns and **names the omitted chapter** rather than silently downloading a hole.

**Duplicate detection fingerprints pages 2–4, never page 1.** This is measured, not a preference:

| pair | truth | page 1 | pages 2–4 |
| --- | --- | --- | --- |
| `144` / `144.0` | same chapter twice | 88 bits apart | 0 of 768 |
| `150` / `150.0` | same chapter twice | 3 bits apart | 0 of 768 |
| `175.1` / `175.2` | two different chapters | **0 (identical)** | 337 of 768 |

Page 1 is wrong in both directions: a re-upload can start at a different opener (so it *misses*
real duplicates), and distinct chapters can share a cover (so it *invents* them). The first version
of this hashed page 1 with exact string equality and silently skipped `175.2` — four real pages
gone. Do not "optimise" the fingerprint back to one page; the extra requests are the price of not
dropping chapters. Empty or different-length fingerprints never match, deliberately: a false
duplicate drops a real chapter, a missed one only costs bandwidth.

The chapter list is **one** AJAX POST (`X-Requested-With: XMLHttpRequest`); a plain GET returns
`0`. No browser, no cookies, no JS. Never resurrect the old next-link walk — it stopped when a
chapter had no images, which is exactly what a Cloudflare challenge looks like, so a blocked run
reported success.

Legacy page trees (`c_7_1.jpg`, unpadded) must be **archived, never deleted**, into the gitignored
`unused/`. Their names collide with ordinal names (`c_000007_*`), so reading them together would
merge two chapters. The downloader refuses to run over one and prints the `mv` command.

On Windows, `-c 144,144.0` must be **quoted** (`-c "144,144.0"`): PowerShell eats the comma as an
array separator, argparse keeps only `144`, and the run silently downloads half the selection.
`-c` here takes *site labels*; `summarize.py -c` takes *ordinals*. Different namespaces.

## Hardware constraint (important)

Target machine has an **RTX 3070 with 8GB VRAM**. Only propose/test models that fit in 8GB with
4-bit quantization. Causal models are loaded with `torch_dtype=torch.bfloat16`, so anything
larger than ~4B params will OOM unless its config entry enables 4-bit quantization. Do not
suggest models the user cannot run locally.

## Scope / disregarded files

- Only work with `novels/the-beginning-after-the-end/`.
- `novels/hunt/` and everything under `summaries/` are **disregarded** — both the source text
  and the previously generated summaries are considered low quality. Do not read, reuse, or
  treat them as reference.

## Known landmines (verify before trusting)

- `src/web_api.py` is **stale and broken**: it imports `local_novel_summarizer` (the file is
  `summarizer.py`) and calls `LocalNovelSummarizer(novel, model, group_size)` where the third
  arg is actually `config`. It also reads `master_summary.json`, which nothing currently writes.
- `generate_all_summaries` in `summarizer.py` no longer writes `master_summary.*` (that block is
  commented out), so anything depending on it is dead.
- ~~`summarizer_config.json` model lookups are indexed directly and an unrecognized `--model`
  raises `KeyError`~~ — fixed: both `summarizer.py` and `summarize.py` resolve entries through a
  helper that names the configured models. The `"default"` block is still never used as a
  fallback.
- ~~`--group-size` and `--batch-size` are parsed but never applied to the summarizer~~ — still true
  of `summarizer.py` (the transformers path), but `summarize.py` now honours `--group-size`. Its
  `--batch-size` equivalent does not exist; batches are sized by `digest.batch_size` plus measured
  token counts.
- `src/git.txt` is a stray diff dump, not config.
- Scraping relies on anti-bot evasion against third-party sites — keep that in mind before
  publishing or automating at scale.

## Testing

`uv run pytest tests` covers the summarizer: chapter selection and resume, context budgeting,
prompt-set resolution, `llama-server` lifecycle, and the batching/geometry helpers behind
`--group-size`, `--digest` and `--all`. It also covers the comic downloader: label selection and
ordinal pinning (`test_comic_selection.py`, `test_comic_manifest.py`), fingerprinting and the
duplicate rules (`test_comic_images.py`, `test_comic_scraper.py`), and filename/grouping
(`test_comic_naming.py`). Everything else (scraping, OCR, real model output) is verified manually:
run a script against a small chapter range and read the output. Prefer adding small unit tests for
the parsing helpers (`parse_chapter_selection`, `parse_chapter_input`, `chunk_text`) when changing
them.

Unit tests cannot catch the failures that matter most here. Every serious defect in the comic
scraper was found by a **live** run on a handful of chapters, not by the suite: the page-1
fingerprint was validated by tests and still dropped a real chapter, and a Cloudflare-shaped hole
looked identical to EOF. So after touching the scraper, run a small live range and read the output.
Two runs are worth the requests: `-c 175-175.2` covers fractional labels, a shared opener and the
`175.2` regression, and `-c "144,144.0"` covers true-duplicate detection (the pair is *outside*
the first range, so that run says nothing about duplicates). Remember to quote the comma.

