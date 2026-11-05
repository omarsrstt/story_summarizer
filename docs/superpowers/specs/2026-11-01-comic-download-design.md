# Comic download stage — design

- **Date:** 2026-11-01
- **Status:** draft for your review (nothing built yet)
- **Author:** agent, from a brainstorming session with the project owner

## 1. Intent / why

`src/comic_reader.py` already turns comic pages into chapter text and feeds
`summarize.py` unchanged. What it cannot do is **get the pages**. `src/comic_scraper.py`
exists to fill that gap but does not work: on real chapters it downloads nothing and
reports success.

The owner's goal is a download stage that is *correct first*: it must never report a
clean finish while having saved nothing, and it must produce a corpus the rest of the
pipeline can read without special cases. It should also reuse the existing scraper
stack rather than reimplement it, because the novel scraper has already absorbed this
site family's quirks (Cloudflare, retry policy, chapter selection).

## 2. What we measured (evidence, 2026-11-01)

All numbers come from probing the live series list (300 entries) and page HTML.

| Finding | Evidence |
|---|---|
| The chapter list is one request | `POST /manga/<slug>/ajax/chapters/` with `X-Requested-With: XMLHttpRequest` returns all 300 chapters (~84 KB). Without that header it answers `0`. The previous docstring's "the endpoint answers 0" is true only of the unheadered request. |
| Image filenames have ≥8 shapes | `c_N_N.jpg` (200 chapters), `ch_N_N.jpg` (70), `c_N_N.webp`, `ch_N_N.webp`, `N.webp`, `N-(N).jpeg`, `N-(N).webp`, `ch-N-img-N.jpg`, `N.png`. 16 chapters mix shapes internally. |
| The current regex rejects 100 of 300 chapters | `IMAGE_NAME` matches only `c_N_N.jpg`. Real run: `-c 216 --dry-run` printed 81 × `skipping: Not a chapter image name: '67.webp'`, then "0 to download", exit 0. Chapter 175 (`ch_174_1.jpg`) behaves the same. |
| The CDN storage number is **not** a chapter id | 285 chapters have a numeric `chapter_<n>` dir, 15 have a hash dir. The number repeats across different chapters: `chapter_147` holds both *Chapter 145* (89 pages) and *Chapter 144.0* (80 pages); 3-way collisions exist at 273/274/275. Zero URL overlap between colliding chapters, so they are distinct content. |
| Fractional labels are **not** stubs | 26 fractional entries. `175.1`–`175.19` are 18 real chapters, mostly 58–79 pages (smallest: `175.2` = 4, `175.5` = 9). `188.5` = 2, `225.5` = 1. |
| Two pairs are duplicate re-uploads | `144`/`144.0` and `150`/`150.0`: per-page average hashes are identical (hamming 0/256) on pages 2–10. Distinct filenames, same art. `144.0` is shorter (80 vs 81 pages), so the re-upload can drop a page. |
| Chapters above the list's max return 200 with no images | `chapter-281` … `chapter-999` all return HTTP 200 and **zero** `img.wp-manga-chapter-img`. "No images" is the only reliable end-of-series signal; HTTP status is not. |
| Chapters 260–265 are absent entirely | Not in the list and not reachable by slug. A gap in the source, not a scraper bug. |
| Site chapter N is stored as `chapter_<N-1>` for the early run | Chapter 1 → `chapter_0/c_0_1.jpg`; chapter 2 → `chapter_1/c_1_1.jpg`. The drift is not constant further into the series (chapter 280 → `chapter_284`). |
| The site list comes back **newest-first** and is in reading order when reversed | The raw response begins `chapter-280-237` and ends `chapter-1`. Reversed, it reads `…174, 175, 175.1 … 175.19, 176, 177…`. |
| Reading order is **not** the numeric label order | Sorted numerically, `175` would come before `175.1`; the site puts `175.1` first. So list position is authority, not the number. `175.1`–`175.19` were inserted between 175 and 176 after the fact. |
| The list needs no cookies and no browser | A cold `curl_cffi` session with only `X-Requested-With` returned 200 and all 300 entries. No `cf_clearance`, no `undetected_chromedriver`. |
| Page 1 is **not** shared boilerplate | Average-hash distance between page 1 of distinct chapters: 104–161 of 256. Only pages of the same art score ≈0. ⚠️ **This row is wrong and is kept as a record of the error — see §7.6.** The method measured *distinct* chapters and never tested *duplicate* ones. Both real duplicates differ on page 1 (88 and 3 bits), and two distinct chapters were measured byte-identical there (`175.1`/`175.2`). Page 1 is the one page that must not be hashed. |

## 3. Goals

1. Download every chapter the site actually has, in all eight filename shapes.
2. Never report success for a chapter that saved no pages.
3. Names on disk must be unique per chapter and sort into reading order.
4. Keep `comic_reader.py` and `summarize.py` **unchanged**: both key on a plain integer
   ("`chapter_0*(\d+)\.txt`" and `group_chapters`' int map) and must keep working.
5. Reuse the existing scraper stack instead of copying it.
6. Leave the pipeline resumable and re-runnable.

## 4. Non-goals

- Reworking the summariser or its prompt/config surface.
- Multi-site support. The comic scraper targets one site; only its URL shape is
  parameterised.
- OCR quality. That is `comic_reader`'s problem and is already documented.
- Fixing the 260–265 source gap.

## 5. Anti-goals (a failure even if it "works")

- A run that saves nothing and prints a clean finish.
- Two chapters silently sharing a filename so one overwrites the other.
- Losing a real chapter because its label looked unusual.
- Downloading a duplicate re-upload twice and letting the summariser recap the same
  story beat twice.
- Fetching anything off-site, or executing page JavaScript.

## 6. Decisions taken with the owner

- **On-disk chapter number is an ordinal**, not the site label. The owner chose this
  over putting the label in the filename because it requires no change to
  `comic_reader.py` or `summarize.py`, and cannot collide.
  - Consequence accepted: `-c N` in `summarize.py` means "the Nth chapter read", not
    "site chapter N". The manifest keeps the mapping, and the comic scraper's own
    `-c` takes site labels (§7.4) and converts them.
- **Duplicate re-uploads are skipped by content hash and reported**, not merged
  silently and not kept as separate chapters (which would make the master summary
  see the same story twice).
- **The 7 existing chapters are re-downloaded and re-read** under the new scheme,
  because they are currently keyed to storage numbers and `chapter_00000.txt` is
  really site chapter 1.
- **The old artifacts are archived, not deleted.** `comics/TBATE`,
  `novels/comic-TBATE`, `transcripts/comic-TBATE` and `summaries/comic-TBATE` move to
  `unused/2026-11-01-pre-ordinal/` (already gitignored) before the first ordinal run.
  Nothing is lost and the fresh tree contains only ordinal-named data.

## 7. The how

### 7.1 Chapter list: one request, not a walk

Replace the next-link walk with a single AJAX fetch
(`POST <series>ajax/chapters/` with `X-Requested-With: XMLHttpRequest`), parsed with
BeautifulSoup (the same parser `scraper_parse`/`scraper_session` already use). ~~The walk
stays only as a fallback if the AJAX list comes back empty.~~ **Built differently, and
deliberately: there is no fallback.** An empty or unparseable list raises instead
(`src/comic_scraper.py:fetch_chapter_list`). A walk that silently substitutes for a failed
list is the failure mode being removed — it cannot tell "the series ended" from "the request
was blocked", so it would report a truncated download as complete.

No browser and no harvested cookies are needed for this (measured), so the download
stage keeps the simple strategy the current script uses: plain `curl_cffi` through
`scraper_fast.default_transport`, with `scraper_parse.looks_like_challenge` to tell a
challenge from the end of a series. `scraper_session.FastSession` and
`undetected_chromedriver` are deliberately **not** pulled in: the whole point is that
no JavaScript runs and no visible window opens. If a challenge ever does appear, the
run stops with the existing "wait and retry, or harvest with a browser" message rather
than silently continuing.

**Order.** The response is newest-first, so ordinal 1 is the **last** entry. Reversing
the list is the reading order; the numeric label is not, because the site inserted
`175.1`–`175.19` between `175` and `176` after the fact. Verified against the raw
response, not assumed.

### 7.2 Manifest: `comics/<name>/chapters.json`

The scraper writes the mapping it discovers, because nothing else can reconstruct it:

```json
{
  "series": "TBATE",
  "site": "…/manga/the-beginning-after-the-end-series/",
  "chapters": [
    {"ordinal": 1, "label": "Chapter 1", "number": 1.0, "slug": "chapter-1",
     "pages": 1, "content_hash": "…", "duplicate_of": null}
  ]
}
```

`ordinal` is the reading-order position (1-based). `label` is the site's own text, so
provenance survives into every transcript and summary. `duplicate_of` names the
ordinal whose art this chapter repeats.

**The manifest is the authority, not a snapshot.** Chapters are matched to ordinals by
`slug`, which is the only stable identifier the site offers. A slug already in the
manifest keeps its ordinal forever; a newly seen slug is inserted at the position the
site's list gives it relative to known neighbours, and **later ordinals do not
shift**. That keeps resume-by-ordinal honest and means a site-side insertion can never
renumber chapters into filenames that already exist on disk. If an insertion lands
between two existing ordinals, the new chapter takes the next free integer after the
preceding one and is flagged in the manifest, so a gap in ordinals is possible and is
visible rather than silent.

**Duplicates are recorded and skipped, keeping their slot.** A chapter detected as a
re-upload gets an ordinal and `duplicate_of: <canonical ordinal>`; no page files are written.
~~`pages: 0`~~ — built differently: the record keeps the page count the scan measured, because
that number is evidence about the site and blanking it would destroy the only record of what the
duplicate was. Ordinals therefore stay dense over the site's list, and reading shows a deliberate
one-chapter jump rather than a numbering surprise. When a pair collides, the chapter **with more
pages wins** (the re-upload can lose a page: `144` has 81, `144.0` has 80), and the loser is the
one marked `duplicate_of`.


### 7.3 Filenames

Pages keep the `c_<chapter>_<page>.<ext>` shape the reader already parses, with the
**ordinal** in the chapter slot, zero-padded: `c_000012_003.jpg`.

- Zero-padding keeps lexical order equal to reading order, and the extension preserves
  the source format (`.jpg`, `.png`, `.webp`).
- `comic_reader.CHAPTER_PATTERNS[0]` already accepts any digit run, so the reader needs
  no change. It opens pages by path, so webp/png need no code change either.
- **Legacy files must be moved out of the way first.** The 7 existing chapters are named
  `c_0_1.jpg`, `c_1_1..c_1_54.jpg`, … `c_6_*`. Those match the *same* reader pattern, so
  a re-download alongside them would make `group_chapters` see chapter 1 as 54 legacy
  pages plus the new pages, merged, with duplicate page numbers — silent corruption.
  The download stage therefore **refuses to run** when the output directory holds page
  files that are not in manifest-ordinal form, and prints the quarantine command
  instead of guessing. Per the owner's decision the legacy trees are archived to
  `unused/2026-11-01-pre-ordinal/`, never deleted automatically.

### 7.4 Reuse audit — replace copies with imports

| Duplicated today | Reuse instead |
|---|---|
| `load_config` (byte-identical to the reader's) | `comic_reader.load_config` |
| `IMAGE_NAME` regex | one shared pattern table in `comic_reader` (it owns chapter/page parsing) |
| retry/backoff constants and loop in `image_bytes` | `scraper_fast`'s `_MAX_ATTEMPTS` / `_BACKOFF_BASE` / `_RATE_LIMIT_*` policy, factored into one helper both call |
| `parse_chapter_page`'s regex HTML scraping | BeautifulSoup, as `scraper_parse` does |
| chapter selection | `scraper.parse_chapter_input` for plain selections |

**Selection cannot be fully reused.** `scraper.parse_chapter_input` parses integers, so
`175.1` and `175.1-175.19` raise `ValueError`, and 26 labels are fractional. The comic
stage keeps that integer parser for ordinals and adds a small label parser of its own
(fractional-aware, accepting `175.1`, `175.1-175.19`, and ordinals), because site
labels are what a user reads off the site. **`-c` therefore accepts site labels**
(`-c 175.1-175.19`), and `--ordinals` selects by manifest position instead.

### 7.5 Failure reporting

Per chapter, record pages wanted / saved / failed. A chapter with a gap is reported in
the end-of-run summary and the exit code is non-zero. "Downloaded 0 pages" can never
look like success.

The exit code tests **pages present on disk after the run**, not bytes written during
it, so a fully resumed run (everything already downloaded) exits 0. A per-chapter
failure, or a chapter that ends up with zero pages on disk, is a failure.

### 7.6 Duplicate detection

> **Superseded 2026-11-02 — do not implement this as written.** The rule below was
> implemented, shipped, and silently dropped a real chapter (`175.2`, 4 pages) on the first
> live run. The premise in the last sentence is false. Keep reading for what replaced it.

Compute a 16×16 average hash (PIL + numpy, already in the `comic` extra) for the
chapter's pages and compare against chapters already indexed; a per-page distance
≤10/256 is the same picture. ~~**Hash from page 1**, which was verified not to be shared
boilerplate (distinct chapters differ by 104–161/256).~~ A chapter whose pages match
another's is marked `duplicate_of`, printed with both labels, and not saved.

**What actually works, and why page 1 fails in both directions.** Measured on the live site:

| pair | truth | page 1 | pages 2–4 |
| --- | --- | --- | --- |
| `144` / `144.0` | same chapter twice | 88 bits apart | 0 of 768 |
| `150` / `150.0` | same chapter twice | 3 bits apart | 0 of 768 |
| `175.1` / `175.2` | two different chapters | **0 (identical)** | 337 of 768 |

A re-upload can begin at a different opener, so page 1 *misses* real duplicates. Two different
chapters can share a cover, so page 1 *invents* them — that is what dropped `175.2`. The
replacement hashes **pages 2–4** into one 768-bit fingerprint and compares by hamming distance
(`DUPLICATE_DISTANCE` per hashed page, so 30 of 768), which separates every measured pair with
room to spare. Page 1 is not fetched at all. Empty and different-length fingerprints never
match, deliberately: a false duplicate drops a real chapter, a missed one only costs bandwidth.
See `src/comic_images.py:chapter_fingerprint` and `src/comic_scraper.py:_is_duplicate`.

The methodological lesson, since this file is the record: measuring whether *distinct* things
differ says nothing about whether *duplicates* match. Both must be measured.

The download stage needs both extras for the *scan* only, because the hash lives in
`comic_images` (PIL + numpy). ~~so it is declared as requiring `uv sync --extra comic --extra
scraper` and fails with that exact message rather than a bare import error.~~ Built differently:
the imports are deferred into the functions that hash, so a `--extra scraper`-only install can
still download without the OCR extra, and the reader is what states the `uv sync` command
(`src/comic_reader.py`). Keeping the download path importable on the smaller install was worth
more than an early, friendly error.

## 8. What is left to the builder

- Exact hash threshold tuning if a false match ever appears (16×16, ≤10/256 was
  measured clean; the margin to real chapters is an order of magnitude).
- Whether the manifest is rewritten atomically per chapter or once at the end. (Per
  chapter is safer; a long run should not lose the mapping.)
- The precise name and shape of the quarantine flag from §7.3.
- Error text and progress formatting.

## 9. Verification

- Unit tests: filename parsing across the 8+ real shapes; ordinal assignment from the
  newest-first list; ordinal stability when a chapter is inserted; duplicate marking;
  refusal to save a non-image; refusal to run over legacy filenames; a chapter with 0
  saved pages reported as failure; a fully resumed run exiting 0; fractional `-c`.
- Manual: download chapters 1–3 and 175–175.2, confirm page counts match the site,
  confirm `comic_reader.py` groups them into the expected chapters, and confirm
  `summarize.py --dry-run` lists them in reading order.
- Regression: `uv run pytest tests` stays green.
