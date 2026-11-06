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

The download step needs the site configured once, the same way the novel scraper does:

```bash
cp src/comic_site.example.json src/comic_site.json   # then fill in the two fields
```

`src/comic_site.json` is gitignored; everything else the comic pipeline needs is tracked in
`src/comic_config.json` (glossary, tile geometry, output dirs). Transcripts are written to
`NOVEL_DIR` from `summarizer_config.json` — the same setting `summarize.py` reads — so the two
stages cannot disagree about where they are.

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
It is the only stage that talks to the site, and it is small: one request for the chapter list,
one request per chapter page, then the images.

```bash
uv run python src/comic_scraper.py --slug the-beginning-after-the-end-series -n TBATE --dry-run
uv run python src/comic_scraper.py --slug the-beginning-after-the-end-series -n TBATE --all
uv run python src/comic_scraper.py --slug the-beginning-after-the-end-series -n TBATE -c 1-3,175.1-175.3
```

   - `-n`: local folder under `comics/`. `--slug` is the series slug from the site URL and
     defaults to `-n`; it exists because the folder name and the URL are not always the same.
   - `-c`: **site labels**, e.g. `1-5,8` or `175.1-175.19`. This is not `summarize.py -c`,
     which takes ordinals — see below. A range spans the chapters *between* its endpoints in
     reading order, which is the only honest reading of `175.1-175.19`: `175.10` and `175.1` are
     the same number but different labels, so counting in floats cannot name that set. A label
     need not be numeric (`175.5a`, `280 - 237`) — it is matched as text. A selection that is
     neither a label nor the word `all` on its own is refused (`-c chapter` is a typo, not "every
     chapter"); `--all` is how you ask for everything.
   - `--ordinals`: positions in `chapters.json` instead of labels, e.g. `1-5,8`. Use this once
     the manifest exists and you want "chapter 144 of the download" rather than "Chapter 144".
   - `--from` / `--to`: a label range, inclusive, e.g. `--from 175.1 --to 176`.
   - `--all`: every chapter in the list. Mutually exclusive with `-c` and `--ordinals`.
   - `--overwrite`: re-fetch pages that already exist.
   - `--dry-run`: print the resolved plan — count, first and last chapter, every ordinal — and
     stop. It does **not** scan, which is the point: a scan is a request per chapter and answers
     a question you did not ask.
   - `--delay`: seconds before each page request (default `0.4`).

**The chapter list is one request.** A plain GET of the site's AJAX endpoint answers `0`; a POST
carrying `X-Requested-With: XMLHttpRequest` returns all ~300 chapters at once. Earlier versions
walked the next-chapter links and stopped when a chapter had no images — which is also what a
Cloudflare interstitial looks like, so a blocked page could end a download early and report
success. That walk is gone, and a challenge page now raises instead.

**The manifest, and why ordinals are pinned.** `chapters.json` in the output folder records each
chapter's `slug` (the site's identifier), `label`, and `ordinal` — its position in reading order.
The ordinal is assigned once and kept forever. Re-scraping months later can therefore add a
chapter without renumbering the ones already downloaded, which matters because the site inserts
chapters retroactively (all of `175.1`–`175.19` arrived between `175` and `176`). Reading order is
the site's list order, not the numeric order of the labels: `175.1` comes *before* `175` in the
list.

Because ordinals are pinned, a retroactively inserted chapter is appended at the end — so a manifest
can be out of step with the site's order. A range is still resolved by position, but if it walks past
a chapter the site places inside it, the run **prints a warning naming the omitted chapter(s)** and
then downloads the span. It warns rather than refuses because "1 through 6" is a servable request;
it does not stay silent because a range that quietly drops a chapter is the one failure this design
exists to prevent.

**Filenames use the ordinal, not the site's number.** Pages are written as
`c_<ordinal>_<page>.<ext>`, six digits on the ordinal and three on the page —
`c_000012_003.jpg`. Zero-padding makes lexical order equal reading order, which is what lets
`comic_reader.group_chapters` sort by name alone. It is deliberately *not* the CDN's
`chapter_<N-1>` directory number: the site files chapter N under `chapter_<N-1>`, but a re-upload
can publish the same number twice (`Chapter 144` and `Chapter 144.0` are two chapters, both 144.0),
so the number cannot identify a chapter. The ordinal can.

**Each chapter gets its own folder.** Pages live at
`comics/<name>/c_000147/c_000147_007.jpg` — one directory per chapter, filename unchanged. A full
series is roughly 15,000 files, and one flat directory that size is legal but painful to browse and
impossible to re-download a single chapter from. Keeping the full filename means a page copied out
of its folder still says which chapter it belongs to. `comic_reader` reads both this layout and the
older flat tree, so an interrupted migration cannot strand a chapter.

**Legacy trees must be archived first.** Pages written before ordinals existed look like
`c_7_1.jpg` — an unpadded chapter number. They match the same reader pattern as the new names, so
reading them side by side would merge two chapters' pages (chapter 7's old pages and its new ones
would both claim `c_000007_*`). The download refuses to run over such a tree and prints the
command; move the old tree into the gitignored `unused/2026-11-01-pre-ordinal/` by hand. Nothing
is ever deleted automatically.

**Duplicates are settled before anything is downloaded.** Before downloading, the run scans each
selected chapter: one request for the chapter page (page count) and up to three for **pages 2–4**,
each reduced to a 16×16 average hash and concatenated into one 768-bit fingerprint. Two chapters
are duplicates when their fingerprints are within 30 bits — ten bits of slack per hashed page, for
a re-encode. Within a group the chapter with **more pages wins** (the re-upload can lose a page —
`144` has 81, `144.0` has 80), ties go to the earlier ordinal, and the loser is marked
`duplicate_of` and skipped. That is four requests per chapter up front — roughly 1200 for a
300-chapter series — and it is deliberate: it is how the duplicate question is answered once,
before any bytes are written, rather than after downloading both copies. The flags are
**re-derived every run** from a fresh scan and saved back to `chapters.json`; a stale value in the
file is never trusted, because the question is about the pages as they are now.

**Page 1 is never hashed, and never fetched.** It looks like the obvious fingerprint and is wrong
in both directions, measured on this series:

| pair | truth | page 1 | pages 2–4 |
| --- | --- | --- | --- |
| `144` / `144.0` | same chapter twice | 88 bits apart | 0 of 768 |
| `150` / `150.0` | same chapter twice | 3 bits apart | 0 of 768 |
| `175.1` / `175.2` | two different chapters | **0 (identical)** | 337 of 768 |

A re-upload can start at a different opener, so page 1 *misses* real duplicates. And distinct
chapters can share a cover, so page 1 *manufactures* them — the first version of this did exactly
that and silently skipped `175.2`, four real pages, which is the failure this whole stage exists to
prevent. Pages 2 onwards separate the two cases completely. A side effect worth knowing: a chapter
too short to have three pages hashable gets a shorter fingerprint, and a shorter one never matches
a longer one. A **one-page** fingerprint is not even offered as evidence — a chapter needs two
hashed pages before it can be called a duplicate at all. That is deliberate. A false duplicate
drops a real chapter; a missed one only costs bandwidth, so every ambiguous case falls on the
cheap side.

**On Windows, quote a comma-separated `-c`.** PowerShell treats `,` as its own array separator, so
`-c 144,144.0` arrives as two arguments and argparse keeps only `144` — it downloads half your
selection and says nothing. Write `-c "144,144.0"`.

**`-c` here and `-c` in `summarize.py` are different namespaces.** The download stage takes *site
labels* (`-c 175.1`), because those are what you read on the site. `summarize.py` takes *ordinals*
(`-c 5`), because by then the chapter is a local file. Pass the wrong one and you get a loud
error naming the label it could not find; when the selection is a bare number it also suggests
`--ordinals`. Note that `-c 144` and `-c 144.0` select **different chapters**.

Only the host in `comic_site.json`'s `comic_site_host` and its subdomains are fetched, over
HTTP(S) only; the pages inject obfuscated scripts that redirect to ad hosts, so off-site image URLs
are refused and no JavaScript is ever executed. `comic_series_url` holds the series URL template,
so the site's address lives in one place rather than in the code. That file is gitignored, exactly
like `scraper_config.json`: copy `src/comic_site.example.json` to `src/comic_site.json` and fill it
in. The downloader refuses to start without it; the host is read once per run and falls back to a
placeholder that matches nothing if the file is missing or malformed, so a config typo makes the
check permissive of nothing but never raises mid-download.

The download exits non-zero if any chapter failed or ends with zero pages on disk, and the exit
code reports what is *on disk*, not what this run fetched — so a fully resumed run is a success.

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

Short fused words need a different test from long ones. A run of 7+ letters is assumed fused; a
shorter run is ambiguous, because `HEHAS` (`he has`) and `SAPIN` (a place) are the same shape. Short
runs are therefore split only when **every part is a common English word** — `HEHAS` (cost 5.8),
`IWAS` (6.6), `TOFORM` (8.0) and `ABIT` (9.6) qualify, while `ETISTIN` (10.5), `SAPIN` (11.7) and
`ELENOIR` (13.8) do not. The 10.0 cut-off sits in a measured gap on the real corpus, not a tuned
point. A run that is already a word is never split (`MANA`, `ALICE`), a contraction tail is never
invented (`THEYRE` stays whole rather than becoming `they re`), and a suffix is never shaved off a
stem (`AUGMENTER`, not `augment er`).

Proper nouns are the limit of what a dictionary can infer: nothing in the text distinguishes
`sap in` from `block him from`. `src/comic_config.json` therefore carries a per-series `glossary`
of names and story terms; add to it whenever a recap mangles a name. The same file holds the tile
geometry. It is tracked and holds nothing site-specific — the site's address lives in the
gitignored `src/comic_site.json` instead.

The glossary is an aid, not a measure of quality. Most comics have no source novel, so a fix that
only works because names were supplied is not a fix — judge a change by how it performs with an
empty glossary. Quality can still be tracked without one by counting tokens that are neither
dictionary words nor glossary terms: on comic chapter 2 the short-run repair above took that from
5.2% to 3.4% and removed every fused-word token.

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