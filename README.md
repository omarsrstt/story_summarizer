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

1. Clone the repository:
   ```bash
   git clone https://github.com/omarsrstt/story_summarizer.git
   cd story_summarizer
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the `config.json` file with the appropriate website details and selectors.

## Usage
### Scraping Chapters
To scrape chapters from a website, run the `scraper.py` script:

```bash
python scraper.py --website <website_name> --novel <novel_name> --chapter <chapter_number>
```
   - `website`: The name of the website to scrape from (as defined in config.json).
   - `novel`: The name of the novel to scrape.
   - `chapter`: (Optional) The specific chapter number to scrape. If omitted, all chapters will be scraped.

### Summarizing Chapters
To summarize the scraped chapters, run the `summarizer.py` script:

```bash
python summarizer.py --novel <novel_name> --model <model_name> --group-size <group_size> --chapters <chapter_selection>
```

   - `novel`: The name of the novel to summarize.
   - `model`: (Optional) The NLP model to use for summarization (default: meta-llama/Meta-Llama-3.1-8B-Instruct).
   - `group-size`: (Optional) The number of chapters to group together for summarization (default: 5).
   - `chapters`: (Optional) Specific chapters or ranges to summarize (e.g., 1,3-5,7).