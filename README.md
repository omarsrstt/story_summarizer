# Story Scraper and Summarizer

A Python-based project to scrape stories from the web and generate concise summaries of each chapter using NLP techniques. This tool is designed for readers who want to quickly grasp the essence of a story

## Overview

This project aims to:
1. **Scrape story chapters** from websites that host them.
2. **Summarize the chapters** using pre-trained NLP models.
3. **Save the summaries** for easy reading or further analysis.

The project is designed to be lightweight and efficient, making it suitable for users with limited hardware resources.

## Features

- **Web Scraping**: Extracts chapters from websites using BeautifulSoup or Scrapy.
- **Text Summarization**: Generates concise summaries of each chapter using Transformers/NLP models from Hugging Face.
- **Structured Storage**: Saves scraped chapters and summaries in a structured format.
- **Optional Web App**: Can be deployed as a web app for browsing and reading summaries.
- **Batch Processing**: Supports batch processing of chapters for efficient summarization.
- **Customizable Grouping**: Allows users to specify how many chapters should be grouped together for summarization.
- **GPU Optimization**: Utilizes GPU acceleration for faster processing when available.
- **Novel Overview**: Generates an overall summary of the entire novel based on individual chapter summaries.

## Installation

1. Clone the repository and open the project directory:
   ```bash
   git clone <repository-url>
   cd story_summarizer
   ```
2. Install the required dependencies:
   ```bash
   uv sync
   ```
   This installs the summarizer's core dependencies. For the scraper, web app and comic reader,
   add the extras: `uv sync --extra scraper --extra web --extra comic`.
3. Copy `src/scraper_config.example.json` to `src/scraper_config.json` and fill in the target
   site's URLs and CSS selectors. The real config is gitignored, so site details stay local.

## Usage
### Scraping Chapters
Two engines are available. The **fast** engine (default) passes Cloudflare once with an
off-screen browser, caches the resulting cookie and the site's chapter list, then fetches each
chapter over plain HTTP. The browser never becomes visible and never steals focus, and existing
chapters are skipped, so it is safe to run repeatedly (e.g. on a schedule) to pick up new
chapters. The **legacy** engine is the original Selenium/undetected-chromedriver scraper, kept
as a fallback.

```bash
# Fast engine (default)
uv run python src/scraper.py --website <website_name> --novel <novel_slug> --chapter_num <selection>

# Legacy engine (failsafe)
uv run python src/scraper.py --engine legacy --website <website_name> --novel <novel_name> --chapter_num <chapter_number>
```

   - `engine`: `fast` (default) or `legacy`.
   - `website`: The name of the website to scrape from (as defined in the config).
   - `novel`: The novel to scrape. The fast engine expects the site's **slug**; the legacy engine
     expects the search name.
   - `chapter_num`: (Optional) A chapter, list, or range (e.g. `5`, `1,3-5`, or `recent`). If
     omitted, all chapters are scraped.
   - `concurrency`: (Fast engine) Concurrent chapter fetches (default: 3). Keep it modest — the
     site rate-limits aggressive clients.
   - `overwrite`: (Fast engine) Re-scrape and replace chapters that already exist. By default
     existing chapters are skipped.
   - `refresh-cookies`: (Fast engine) Discard the cached Cloudflare cookie/chapter list and
     re-harvest. Use this if a run starts failing with challenges.
   - `delay`: (Fast engine) Seconds to pause before each chapter fetch (default: `0`). Use a
     small value to be gentler on the site. Transient timeouts and 5xx responses are retried
     automatically with backoff.

The fast engine caches its cookie and chapter list under `.scraper_cache/` (gitignored) and
re-harvests automatically when a request is challenged.

The browser's Chrome major version is detected automatically (so it keeps working when Chrome
updates); no manual pin is needed.

#### Harvest modes

Before fetching chapters, the fast engine must learn the chapter list. How it does that is
per-site, set by the `harvest` key in the site's config entry:

| `harvest` | How the list is obtained | Use when |
| --- | --- | --- |
| `"browser"` (default when omitted) | A one-off off-screen browser passes the bot check, then the chapter dropdown and cookies are read out of it | The site challenges plain HTTP requests (e.g. Cloudflare "Just a moment…") |
| `"paginated"` | Plain HTTP walks the site's paginated chapter index | The site serves its chapter list without a challenge — no browser needed at all |

`"paginated"` needs `chapter_list_page_url` and `chapter_list_link_selector`. Pagination is
followed in one of two ways, also chosen by config:

- **Numbered pages** — put a `{page}` placeholder in `chapter_list_page_url`
  (e.g. `https://example.com/novel/{novel_slug}/chapters?page={page}`). Pages are walked
  upwards until one yields no new chapters.
- **Next-link chains** — omit `{page}` and set `chapter_list_next_selector` (e.g.
  `a[rel=next]`) to the link that advances to the following page.

If a paginated site starts challenging requests, the engine stops with an error naming the
site; switch that entry to `"harvest": "browser"` to go back to passing the challenge once
with the off-screen browser. It never silently falls back, so a stale config is visible
instead of mysteriously slow.

Sites that rate-limit bursts (Cloudflare answers with `429` and `Retry-After`) are handled
automatically: 429 responses are retried with backoff, and a rate-limit page is treated as a
hard error rather than as "the chapter list ended", so a partial list can never be mistaken
for a complete one.

#### Chapter file names

Chapters are written as `chapter_<number>.txt` with five-digit padding, so chapter 12 is
`chapter_00012.txt` and the directory sorts correctly in any file browser. The scraper still
recognises two older forms and treats them as already-present, so nothing is ever re-scraped as a
duplicate:

| Form | Status |
| --- | --- |
| `chapter_00012.txt` | current |
| `chapter_0012.txt` | deprecated (written by the legacy engine) |
| `chapter_12.txt` | deprecated (written before padding existed) |

Run `rename-padding` (below) to migrate a directory; the deprecated lookups can be deleted once no
corpus uses them.

### Maintaining the Corpus

`src/corpus_tools.py` tidies an existing chapter directory. Both commands are **dry runs by
default** and print exactly what they would do; pass `--apply` to change files. A backup of the
directory is written beside it (`<dir>.bak`) before the first change, since `novels/` is not
tracked by git.

```bash
# Show what would happen (no writes)
uv run python src/corpus_tools.py all novels/<novel_slug>

# Do it: rename to five-digit padding, then drop site trailers
uv run python src/corpus_tools.py all novels/<novel_slug> --apply

# Or run just one of the two steps
uv run python src/corpus_tools.py rename-padding   novels/<novel_slug> --apply
uv run python src/corpus_tools.py strip-boilerplate novels/<novel_slug> --apply
```

#### How trailers are detected

Sites append furniture to the end of a chapter ("read more at …", "thank you so much!"). That text
changes whenever the site changes, so the tool does **not** contain a list of ad strings. Instead it
learns them: a line is treated as boilerplate when the *same* line ends a large share (≥10%) of the
chapters in the directory. Prose essentially never repeats verbatim across hundreds of chapters,
while site footers do. Consequences worth knowing:

- Only **trailing** lines are considered, and removal stops at the first line that does not recur.
  Dialogue in the middle of a chapter that happens to quote the footer survives.
- Removal is repeated, so a two-line footer is removed one line per pass.
- A chapter is never emptied: if the whole file looks like boilerplate it is left alone.
- Fewer than five chapters gives no statistical signal, so the tool declines to act.

It is deliberately a tool you run, not something applied while scraping, so it never silently edits
text it does not understand.

### Reading Comics

Manhwa and manga need a reading stage the novels do not: the pages have to become text before
anything can summarize them. `src/comic_reader.py` does only that, writing
`novels/<slug>/chapter_NNNN.txt` — the exact layout `summarize.py` already looks for — so recap,
`--group-size`, `--digest` and `--all` then work unchanged.

```bash
uv sync --extra comic --extra scraper
uv run python src/comic_scraper.py --slug my-comic-series -n TBATE -c 1-5
uv run python src/comic_reader.py --series TBATE --comics comics/TBATE --only 1-5
uv run python src/summarize.py -n comic-TBATE -c 1-5 -g 5 --digest
```

   - `--series`: series name; the reader prefixes it to form the slug.
   - `--comics`: directory of page images.
   - `--only`: `all`, `1`, `1-3` (default: all).
   - `--overwrite`: re-read chapters that already have a transcript.

#### Downloading the pages

`src/comic_scraper.py` gets the images into `comics/<name>/` so the reader has something to read.
It reuses the novel scraper's parts rather than its own: `scraper_fast.default_transport` for the
`curl_cffi` TLS impersonation the site's bot check demands (plus timeout/5xx retries and 429
backoff), `scraper_parse.looks_like_challenge`, `scraper.parse_chapter_input` for the `1-5,8`
syntax, and `scraper_store`'s five-digit chapter naming.

```bash
uv run python src/comic_scraper.py --slug the-beginning-after-the-end-series -n TBATE -c 1-5
uv run python src/comic_scraper.py --slug the-beginning-after-the-end-series -n TBATE --all
```

   - `--slug`: series slug from the site URL; `-n` is only the local folder name.
   - `-c`: chapter selection in **site** numbering, e.g. `1-5,8`.
   - `--all`: walk the next-chapter links until the series ends.
   - `--overwrite` / `--delay` / `--dry-run`: as for the novel scraper.

Chapter numbering is the trap here. The site's reader **chapter N** is stored on its CDN as
`chapter_<N-1>`, so chapter 1 is `chapter_0/c_0_1.jpg` and chapter 2 is `chapter_1/c_1_1.jpg`.
Filenames deliberately keep the *storage* numbering, because that is what `comic_reader` groups on
and what a manual save already produces — so a filename can look off by one and still be right.

The chapter list is loaded by JavaScript and its endpoint answers `0` to a plain request, so `--all`
walks the page's own next/prev links instead and stops when a chapter carries no reader images.
That is also why the challenge check matters: a Cloudflare page has no images either, and without
distinguishing them the walk would end early and still report a clean finish. A non-200 chapter
page aborts the run for the same reason. Verified byte-identical against a manual download of
chapters 1 and 2, which is also how a page missing from that manual set (`c_1_38.jpg`) was found.

Only `comic-site.test` and its subdomains are fetched, over HTTP(S) only; the pages inject obfuscated
scripts that redirect to ad hosts, so off-site image URLs are refused and no JavaScript is ever
executed.

#### Comics get their own slug

`--series TBATE` writes to `novels/comic-TBATE/`, so the summariser is then run with
`-n comic-TBATE` and everything lands in `summaries/comic-TBATE/`. The `comic-` prefix is applied
by the reader, because a comic and a novel can share a name — both resolve to `<root>/<slug>`, so
an unprefixed slug would let one overwrite the other's transcripts and summaries. The reader
prints the exact `-n` value to use when it finishes, and re-running with the slug it printed does
not stack prefixes.

Both page layouts work: one image per reader page (`c_1_07.jpg`) and a whole chapter stitched
into one continuous scroll (`c_0_1.jpg`, 45 722 px tall). A scroll is cut into overlapping tiles
because the detector only handles regions roughly its own size, and a bubble on a seam is dropped
when both neighbouring tiles read it.

Pages are **not** upscaled. Measured on real pages, 1× and 2× output were character-identical
while 2× cost 31% more time, because the detector already resizes internally.

Two OCR defects are corrected, since both visibly corrupt a transcript:

- **Lost word spaces.** The recogniser is Chinese-trained and drops Latin inter-word spaces, so
  58% of real blocks arrived fused (`HEDOESN'THIT`). A dictionary segmenter repairs them, guarded
  so it cannot shred names — `SAPIN` stays `SAPIN` rather than becoming `sap in`.
- **Hallucinated CJK glyphs** inside otherwise English text, which also crashed a cp1252 console.

Proper nouns are the limit of what a dictionary can infer: nothing in the text distinguishes
`sap in` from `block him from`. `src/comic_config.json` therefore carries a per-series `glossary`
of names and story terms; add to it whenever a recap mangles a name. The same file holds the tile
geometry.

Filtering uses the segmenter's own 126k-word list rather than length guesses, which on a real
chapter kept 287 of 294 blocks and dropped only page furniture (`213.`, `+`, `/`, `7`). A rare
noise token still gets through, because the word list contains abbreviations like `nih`; tightening
further starts discarding real text, which costs a summary more than one stray line.

Structured transcripts are kept alongside the flat text at
`transcripts/<slug>/chapter_NNNN.jsonl` (page, box, text, confidence, and whether the block sat in
a bubble), so later passes — speaker attribution, non-bubble text — have geometry to work from.

### Summarizing Chapters
Use `src/summarize.py`. It drives `llama-server` (llama.cpp) with a GGUF model instead of loading
weights in-process, which is roughly 6x faster on an 8 GB card and lets the model be unloaded to
free VRAM afterwards.

```bash
uv run python src/summarize.py -n <novel_name>
```

   - `-n`, `--novel`: the novel directory name under `novels/` (required).
   - `-m`, `--model`: model key from `src/summarizer_config.json` (default: `qwen3.5-9b`).
   - `-c`, `--chapters`: `all`, `recent`, `5`, `1-20`, `1,3-5` (default: all).
   - `--overwrite`: regenerate summaries that already exist.
   - `--dry-run`: print the plan without loading the model.

Recaps resume by default: a chapter that already has a non-empty
`summaries/<slug>/summary_chapter_NNNNN.txt` is skipped. A chapter whose prompt would overflow
`n_ctx` is refused rather than summarized from its opening pages only.

#### Wider artifacts
Three additional artifacts are built from the per-chapter recaps — never from the source, so any
of them can be rebuilt at a different width without re-reading the novel:

| Flag | Output | What it is |
| --- | --- | --- |
| `-g N`, `--group-size N` | `summary_chapters_00001-00005.txt` | one summary per N chapters |
| `--digest` | `digest.txt` | `Chapter 7: event. event.` — one or two sentences per chapter |
| `--all`, `--master` | `master_summary.txt` | whole-novel summary, compressed in rounds |

The recaps must be generated **first** — these steps read them rather than the chapters, and a
missing recap in the selected range is an error rather than a silent gap. After that the flags
combine and run in dependency order:

```bash
uv run python src/summarize.py -n <novel_name>                      # 1. recaps
uv run python src/summarize.py -n <novel_name> --digest -g 5 --all  # 2. the rest
```

`--chapters` applies to every step, so `-c 1-20 --digest` digests just those twenty and merges
into any existing `digest.txt` rather than replacing it.

Model behaviour (quantization, generation parameters, paths, and the prompts for each stage) is
configured in `src/summarizer_config.json`.