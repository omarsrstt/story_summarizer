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
   This installs the summarizer's core dependencies. For the scraper and web app, add the
   extras: `uv sync --extra scraper --extra web`.
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

### Summarizing Chapters
To summarize the scraped chapters, run the `summarizer.py` script:

```bash
python summarizer.py --novel <novel_name> --model <model_name> --group-size <group_size> --chapters <chapter_selection>
```

   - `novel`: The name of the novel to summarize.
   - `model`: (Optional) The NLP model to use for summarization (default: `Qwen/Qwen3-4B-Instruct-2507`).
   - `group-size`: (Optional) The number of chapters to group together for summarization (default: 1).
   - `chapters`: (Optional) Specific chapters or ranges to summarize (e.g., 1,3-5,7).

Model behaviour (quantization, generation parameters, and the summary prompt) is configured per
model in `src/summarizer_config.json`.