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

1. `src/comic_scraper.py` — downloads page images to
   `comics/<name>/c_<ordinal>/c_<ordinal>_<page>.<ext>` (one folder per chapter).
   Also writes `comics/<name>/chapters.json`, the **authoritative manifest**.
2. `src/comic_reader.py` — OCRs the pages into `novels/comic-<name>/chapter_NNNNN.txt`, one file
   per chapter, using the same `chapter_%05d` convention the novel scraper uses. It drops
   scan-group credit lines from the flat transcript (see below); the structured
   `transcripts/<slug>/chapter_NNNNN.jsonl` keeps every block, so the decision can be revisited
   without re-reading the images.
3. `uv run python src/summarize.py -n comic-<name>` — unchanged from the novel path; it just sees
   another slug.

So the join between comic and novel stages is **plain integers on disk**. Neither `comic_reader.py`
nor `summarize.py` knows the comic scraper exists, and neither was modified for it. That is why the
manifest pins ordinals rather than trusting site numbers.

### Credit lines are furniture, and are matched by shape

A chapter's first and last pages carry the scan group's credits — a URL, a social handle, and a
`Role: Name` block. They are the page's producer, not its story. `comic_text.is_credit_line` drops
them, and the reader applies it via `keep_credit_mask` before writing the flat transcript.

The rule keys on **structure**, never on credit vocabulary, and that is measured rather than
preferred. "control", "original" and "comic" are ordinary English: a rule matching them deleted
`to control MANA is` and `HAVING THE control of`, both real dialogue. `ORIGINAL.` on page 50 of
chapter 181 is the punchline of `NOT VERY / ORIGINAL.`. The four shapes kept are a URL or domain, an
`@handle`, a line led by a role word (`ComicArtist:Fuyuki23` — the OCR glues the space out), and a
bare name following a role line (`ART BY` / `TURTLEME`). Safety comes from *whole-phrase* role
matching: an earlier draft accepted any line containing a job word, so `ART IS LOOKING` made
`AT HIMSELF IN` look like an artist and `to even control` ate `the bodily functions`.

A **block** rule — once a credit is found, keep dropping short lines — was tried and removed. Every
line of a real credit block is one or two words, but so is `THE MIRROR!`, `RUMBLE` and `VIRION!`,
which it deleted from the real corpus. Credits are not reliably contiguous at the text level, so the
local rule wins. Names split onto their own line are left behind as `noise` warnings — the honest
cost of never deleting story text. The one contextual shape is still guarded twice: the line must be
short *and* contain no dictionary word, because `it is imperative` follows `assistant editors` and a
length-only rule ate it.

`comic_check._credit_findings` uses `keep_credit_mask` too, so the warning describes what the reader
actually does. It previously carried its own marker list (`quality control`, `turtleme`, and a site
name) which had drifted from the reader and also put that site name in tracked code.

**The suspicious-rate is a ratio, so quote the count.** Scrubbing credits took chapter 183 from 6
suspicious tokens to 4 while its rate *rose* from 3.75% to 4.55%: the credits were ordinary English,
so removing them shrank the denominator more than the numerator. Across the 7-chapter corpus the
change was 97 -> 67 suspicious tokens (2.63% -> 2.0%), and every `credit` and `repeat` warning
disappeared. `comic_check` now reports both numbers for that reason.

A fourth script, `src/comic_check.py`, sits beside the reader and validates what it wrote. It is
**novel-free by design** — it reads only the transcripts and the manifest — so it can gate a pipeline
change on any comic. Findings are `fail` (unambiguous: CJK glyphs, control bytes, fused runs the
shipped rule would split, page markers past the images, chapters short of their manifest page count)
or `warn` (a smell that shares its shape with real text: a glossary term split into ordinary words, a
rare non-dictionary token, an OCR repetition loop, credits left in prose). Invention fails on the
unambiguous classes; **loss is measured, never gated**, because without a reference a dropped line and
a cut line produce identical findings. The `fused` finding gates **staleness**, not correctness: it
compares a transcript to the rule as it is now, so re-running the reader after a bad rule change makes
the two agree and hides the regression — that shows up only as a rise in the suspicious rate. Two
false-positive classes were found while building it and are pinned by tests: transcripts only mark
pages that produced text (so a silent page is not a gap), and a fingerprint-confirmed duplicate
(`Chapter 144.0`) is never downloaded by design (so it is not short).

### The novel is a test oracle, never an input

Most comics have **no associated novel**, so the goal is an OCR pipeline whose quality does not
depend on one existing. That makes this a hard rule, not a preference:

* **Allowed:** read the novel to *measure* the pipeline and find generic defects. It is how the
  `_MIN_RUN` bug was found — novel text showed `HEHAS`/`IWAS` surviving as single tokens when they
  were plainly `he has`/`i was`.
* **Forbidden:** let novel text reach a transcript. No copying spellings in, no per-chapter
  correction pass, no runtime lookup. A fix that only works because a novel exists does not
  generalise, and hides the defects a comic-only series would hit.

The per-series `glossary` is a legitimate **user-facing** aid, but it is content, not pipeline
quality: filling it from the novel would make the measurement flattering and prove nothing. Judge a
pipeline change by whether it helps with an **empty** glossary.

Two traps when pairing the trees:

* **Never align by chapter number.** Measured on this comic, the offset drifts: comics 1–3 map to
  novels 1–3 (offset 0), comic 147 → novel 100 (offset 47), comic 181 → novel 138 (offset 43). Align
  by content.
* **Some chapters have no counterpart.** Comics 182 and 183 are sparse bonus chapters whose best
  novel match is indistinguishable from noise (score margin ≈ 0.000). A harness must report those as
  *unmatched*, never as OCR failures.

The transliteration cost is measurable without a novel, which is what makes progress checkable:
count tokens that are not dictionary words and not glossary terms. On comic chapter 2 the
`_MIN_RUN` fix moved that from 5.2% to 3.4%, and every fused-word token disappeared.

### The benchmark: what a novel can measure, and what it cannot

`src/comic_benchmark.py` is the **local** instrument that puts numbers on the oracle rule. It is
read-only by construction (verified by hashing the corpus before and after a run) and the pipeline
never imports it; a test enforces that by reading the source of `comic_reader`, `comic_check` and
`comic_text`.

**Recall is not measurable.** The comic is an adaptation with roughly half the novel's word count,
so a missing sentence is usually a legitimate cut. Reporting coverage would punish the adapter for
adapting. What survives is transcription accuracy on the words that *are* present:

| Output | Means |
| --- | --- |
| `name spelling` | of the names the transcript produced that the novel also knows, how many were spelled the novel's way |
| `misread names` | `transcript -> novel`, one edit apart, with `near=` chapters of distance to the novel's nearest use |
| `glossary gaps` | names spelled correctly that the segmenter *would* split, so they need an entry |
| `adaptation spellings` | the comic and novel differ on purpose — **not** an error |
| `suspicious explained` | of `comic_check`'s suspicious tokens, how many map to a novel name |

Current baseline on the 5 alignable chapters: **95.7% of recognised names exact** (44/46).

Four measured traps, all found on the real corpus, all now guarded by tests:

* **A ratio over every name-shaped token is a lie.** `exact / name_tokens` read 43% because the
  denominator included detector junk (`KWNI`, `EHNH`). The metric is `exact / recognised`, and junk
  is reported separately as `name_unknown`.
* **Distance-1 alone manufactures findings.** Unguarded, it reported `AGB -> Agh`, `NOOI -> NOOO`
  and `RAAH -> Riah` as misread names. All three are wrong: `Agh` appears **once** in 10 MB, `NOOO`
  is 4/4 all-caps, and `Riah` first appears in novel **287** while comic 2 aligns to novel 2. A name
  must now be Title Case *and* used at least twice; the distance is printed so a coincidence is
  visible.
* **The glossary outranks the novel.** The tracked glossary has `ASHBERN`; the novel uses `Ashber`
  39 times and `Ashbern` never. The raw OCR settles it — `Ashbern Town` at **0.96 confidence** — so
  the *comic* spells it that way. An earlier version of this tool called that a "glossary defect",
  which would have corrupted a correct tracked file on the novel's authority. It is reported as an
  adaptation spelling, never as wrong.
* **A "glossary gap" is only a gap if the splitter would split it.** Dropping that condition
  reported 11 candidates, of which 10 needed no entry (`Dawsid`, `Eksire`, `Ack`, `Aww`, `Here's`).
  Only `Goodsky` (`good sky`) needs one.

The pairing between the trees is series knowledge, so it lives in the **gitignored**
`src/comic_pairing.json` (`src/comic_pairing.example.json` is tracked). It is a cross-check, not the
mechanism — alignment is by content, and an entry that disagrees with the measured best match makes
the run **exit 1** so a bad pairing is never reported as bad OCR.

### Config: the site is never tracked

The repo is **public**, so the site this tool targets is deliberately absent from it — this is a
standing rule, not a one-off cleanup, and it has already been violated once:

| File | Tracked | Holds |
| --- | --- | --- |
| `src/comic_config.json` | **yes** | glossaries, tile geometry, `comics_dir`, `transcripts_dir` — no site |
| `src/comic_site.json` | no (gitignored) | `sites` map, or the legacy flat site keys |
| `src/comic_site.example.json` | **yes** | the same shapes as placeholders |

**Several sites coexist; `--site` picks one.** `comic_site.json` accepts either the original flat
pair (`comic_series_url`, `comic_site_host`) or a `sites` map of named entries plus `default_site`.
The flat shape is not deprecated and needs no edit: it resolves to an unnamed site, which is what
every existing install has. `comic_scraper.resolve_site` refuses an ambiguous config by naming the
configured sites rather than picking one, because guessing would download the wrong comic into the
folder the user named and the pages would look plausible.

**The site's *shape* is config too.** Chapter-list URL/method/headers, the chapter-link selector, and
the image selector/attribute were hardcoded to the Madara WordPress theme. They are now per-site
overrides whose defaults reproduce that theme exactly (`DEFAULT_CHAPTER_LIST_URL` etc. in
`comic_scraper.py`). This is what makes a second site a config entry rather than a code change — and
a site that needs no overrides is a site running Madara. `Site.list_url` was checked against the
string the old code built by hand: a missing trailing slash asks a different URL and 404s.

**Glossaries are per series.** A flat `glossary` list cannot serve two comics: the entries are
another series' proper nouns, so the validator would accept the wrong names as known and the
segmenter would protect them. `comic_text.resolve_glossary` resolves
`glossaries[series]` → `glossaries["default"]` → the flat `glossary`, so a series with no entry
inherits the flat list and existing single-series configs keep working. It is one function shared by
`comic_reader`, `comic_check` and `comic_benchmark` **because** they must agree: a term the reader
protects but the validator does not is reported as noise forever. The series key matches with or
without the `comic-` prefix and case-insensitively, since the reader takes `TBATE` while the folder
is `comic-TBATE`.

**Where transcripts go is declared once.** `comic_reader` writes
`<NOVEL_DIR>/<slug>/chapter_NNNNN.txt` using `NOVEL_DIR` from
`summarizer_config.json`, not a copy in `comic_config.json`, because `summarize.py`
reads them from there. Two independent copies of that path stay equal only by luck:
when they drifted, the reader succeeded and the next stage failed on a directory
the user never configured. An explicit `novels_dir` in the comic config still
wins, for deliberately separating comic output.

`comic_reader.load_config` reads the tracked file and shallow-merges `comic_site.json` over it, so
an older single-file config that still carries the site keys keeps working. A *missing* site file is
fine (a reader-only run needs no host); the downloader refuses to start when the resolved site is
still the placeholder and says which file to copy. `comic_scraper.DEFAULT_SERIES_URL`/
`DEFAULT_SITE_HOST` are placeholders on the reserved `.test` suffix: they match nothing, so an
unconfigured run refuses every image rather than fetching from a guess.

Two traps, both already hit:

- **Never use `example.com` as the placeholder.** The host check accepts subdomains of the
  configured host, so `example.com` would also accept `evil.example.com` — the guard's own test
  case. `.test` is reserved and safe.
- **Scrubbing the file is not enough.** The site was in the tracked config *and* in 19 commits of
  history; anyone can read old commits. Publishing required a history rewrite, not just an edit.
  New site-bearing work must therefore never be committed in the first place.

`src/scraper_config.json` follows the same rule for the novel scraper, with
`src/scraper_config.example.json` tracked. Any new scraper stage should get its own example +
gitignored real file rather than putting a host in a tracked config.

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
(`test_comic_naming.py`). The transcript validator has its own suite (`test_comic_check.py`), where
each detector must be provably load-bearing: disable it and a named test has to fail. That rule
found three real defects in the validator itself — a page-gap check built on a false premise, a
fused check that split glossary names, and a noise count that treated every contraction as
non-word — none of which a green suite had caught. It also caught three in the credit filter, all
by deleting real dialogue: a "block" rule that ate `THE MIRROR!`/`RUMBLE`/`VIRION!`, a role test
that accepted any line *containing* a job word (`ART IS LOOKING` → `AT HIMSELF IN` gone), and a
length-only context rule that ate `it is imperative`. None appeared until the scrubber was run over
the real corpus and its removals read line by line, which is the only way to test a filter that
deletes text: assert what survives *and* read what does not. Everything else (scraping, OCR, real
model output) is verified manually: run a script against a small chapter range and read the output.
Prefer adding small unit tests for the parsing helpers (`parse_chapter_selection`,
`parse_chapter_input`, `chunk_text`) when changing them.

The benchmark has its own suite (`test_comic_benchmark.py`) under the same rule, and it earned its
keep during construction: the load-bearing sweep caught an *untested* fidelity floor (setting it to
a billion left the suite green) and a glossary-gap rule that never fired. But the defects that
mattered were found by running it against the 516-chapter novel and reading the output — the fake
`AGB -> Agh` misread, the 43%-that-should-be-96% ratio, and the `ASHBERN` inversion all looked
plausible as numbers. **A benchmark needs its output read, not just its tests run**: a metric that is
green and wrong is the most expensive kind, because it steers the next change.

Unit tests cannot catch the failures that matter most here. Every serious defect in the comic
scraper was found by a **live** run on a handful of chapters, not by the suite: the page-1
fingerprint was validated by tests and still dropped a real chapter, and a Cloudflare-shaped hole
looked identical to EOF. So after touching the scraper, run a small live range and read the output.
Two runs are worth the requests: `-c 175-175.2` covers fractional labels, a shared opener and the
`175.2` regression, and `-c "144,144.0"` covers true-duplicate detection (the pair is *outside*
the first range, so that run says nothing about duplicates). Remember to quote the comma.

