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

The fast engine caches its cookie and chapter list under `.scraper_cache/` (gitignored) and
re-harvests automatically when a request is challenged.

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