# Fast, non-intrusive chapter scraper — design

- **Date:** 2026-10-25
- **Status:** draft for your review (nothing built yet)
- **Author:** agent, from a brainstorming session with the project owner

## 1. Intent / why

The current scraper (`src/scraper.py`, Selenium + `undetected_chromedriver`) is
correct but intrusive and slow: it launches a **headful** Chrome, calls
`focus_window()` (a `minimize_window()` + `maximize_window()` hack), and drives
one chapter at a time by clicking a dropdown, with multi-second `sleep`s between
steps. It cannot run headless (Cloudflare blocks it), so it takes over the
machine — the owner cannot game or work while it runs.

**What the owner wants:** scraping to be a *light, background* job. Ideally
invisible, and faster — without losing the ability to scrape the target site at
all.

## 2. What we measured (evidence, 2026-10-25)

| Approach | Result |
|---|---|
| Plain HTTP (`requests`) | ❌ 403 Cloudflare challenge (`Just a moment…`) |
| Chrome TLS impersonation (`curl_cffi`) | ❌ 403 — block is not merely TLS fingerprint |
| `undetected_chromedriver(headless=True)` | ❌ stuck on the challenge |
| `--headless=new` | ❌ still detected after 45 s |
| **Off-screen headful Chrome → harvest cookies → close browser** | ✅ page loaded, `cf_clearance` obtained |
| **HTTP GET a chapter with the harvested cookies** | ✅ **200, ~12–25k chars, ~0.25 s, no browser** |
| Cookie cache reused in a **fresh process, no browser** | ✅ still authorised |
| Chapter list via HTTP (`select.select-chapter`) | ❌ present but **0 options** — JS-populated |
| Chapter list via off-screen browser (click `glyphicon-list-alt`) | ✅ **533 options** — `value`=slug, `text`=chapter title |

Conclusion: a browser is needed **once per cookie lifetime** to pass Cloudflare;
everything else can be plain HTTP. This is the basis of the design.

## 3. Goals

1. Run **in the background** without stealing focus or covering the screen.
2. Be **fast**: fetch many chapters quickly, optionally in parallel.
3. **Preserve the legacy method** as a fallback for when the fast path breaks.
4. Make the fast path the **default**, documented in the README.
5. Absorb the site's quirks: variable chapter numbering, slug URLs, rate limits.
6. Keep the output **identical in content** to what the legacy scraper produced.

## 4. Non-goals

- Reworking the summarizer.
- Adding new target sites (the site-specific pieces stay in `scraper_config.json`).
- Printing/serving scraped text anywhere beyond `novels/`.

## 5. Anti-goals (this would be a failure even if it "works")

- Scraping at a volume/rate that gets the IP banned or harms the site.
- Silently writing wrong or partial chapters over good local files.
- Reintroducing anything that steals focus or opens a visible window.
- Publishing site names/URLs in committed files (existing constraint).

## 6. Constraints

- Target machine: Windows, with a **recent** Chrome (155 when this was written; 156
  during implementation). The major version is **detected at runtime**
  (`detect_chrome_major()`), never hardcoded, so moving to a new Chrome does not
  break the harvest.
- `undetected_chromedriver` downloads a driver for the *latest* Chrome and
  **fails** unless pinned via `version_main` — so the fast path pins the driver
  version to the detected installed major (or defer to auto-detection if unknown).
- Cookies must handle Cloudflare expiry; a stale cookie returns to the challenge.
- `curl_cffi` is a **new dependency** (proposed as a `scraper` optional extra).
- The repo is public; the real `scraper_config.json` stays gitignored.

## 7. Decided design

### 7.1 Shape

Two engines behind one CLI, default = fast, `--engine legacy` = current code:

- **`src/scraper_fast.py`** — new engine. HTTP-first, optional off-screen browser
  for cookie acquisition only.
- **`src/scraper.py`** — unchanged legacy engine, kept as the failsafe.

### 7.2 Fast-path data flow

1. Load a cached cookie file (`spike_out`-style cache dir; path TBD).
2. If no cache, or a probe request returns the challenge (403), acquire cookies:
   launch **off-screen** headful Chrome at `--window-position=-32000,-32000`,
   load one chapter, wait for the content selector, read cookies + UA, then
   **harvest the full chapter list** (click the `glyphicon-list-alt` icon and read
   `select.select-chapter` — an ordered list of `(value, text)` where `value` is
   the chapter URL slug and `text` is the chapter title). Cache cookies *and* the
   map together, then **close** the browser.
   (Pin `version_main` to the installed Chrome major version.)
3. For each requested chapter: `GET /book/<slug>/chapter-<index>` with
   `curl_cffi` (Chrome impersonation) + cached cookies + matching UA.
4. Parse with BeautifulSoup: content `div.txt`, title `div.top span.chapter`.
5. Save to `novels/<novel>/chapter_<n>.txt` in the existing format.
6. On 403 / challenge mid-run → refresh cookies once, retry.

### 7.3 Speed & politeness

- Default: **small concurrency (3–4 parallel requests)** with a small
  inter-request delay; `--concurrency N` tunes it (use `1` for sequential).
- **Retry with exponential backoff** on timeout/5xx (we observed consistent
  timeouts on some chapter indices and 403s after bursts). **Implemented:**
  `default_transport` retries up to 3 times with exponential backoff (0.5s, 1s)
  on a timeout (status 0) or 5xx; a `--delay S` flag adds an optional pause
  before each fetch.
- No `focus_window()`, no multi-second fixed sleeps.

### 7.4 Chapter discovery & numbering — **needs your call**

Facts:
- Local corpus: `chapter_1.txt … chapter_512.txt` (unpadded) + `chapter_0513 … 0516`
  (padded). 513 files; **510–512 missing**.
- Site chapter index in the URL does **not** map linearly to chapter numbers:
  `/chapter-2`→Ch1, `/chapter-6`→Ch5 (offset +1), `/chapter-100`→Ch98 (offset
  +2), `/chapter-517`→Ch517 (offset 0). `/chapter-1` is a prologue (`c-1`).
- Recent chapters use **slugged URLs** (`/chapter-530-end-story`); enumeration
  should not assume plain `/chapter-{N}`.

Options to choose from:
- **(A)** Save by the **chapter number parsed from the page title**, mapping to
  the local unpadded scheme (safest for matching the existing corpus).
- **(B)** Save by the **site URL index** (simple, but renumbers vs. your files).
- **(C)** Keep both: filename = local number, and store the site index in a side
  map.

**Decided (owner, 2026-10-25): (A).** Save `chapter_<n>.txt` where `<n>` is the
chapter number read from the page/title; the URL index is used only to fetch.
Gap-fills therefore line up with the existing corpus.

**URL resolution (builder call):** the fast engine does **not** compute URLs from
an arithmetic offset (the offset is not constant) and does **not** scrape the
chapter list from the novel page (it only renders range labels, `C.1 - C.40`,
and a handful of links). Instead it builds a *local-number → URL* map from the
`select.select-chapter` list harvested during the browser visit, reading each
chapter's number from its `text` (e.g. `"Chapter 1: The Light At The End of The
Tunnel"`). Unmapped numbers are reported, not guessed. Recent chapters use
slugged URLs (`/chapter-530-end-story`), which the harvested `value` carries
verbatim; the `c-1: Prologue` entry is not one of the numbered corpus chapters.

### 7.5 CLI (proposed)

```bash
uv run python src/scraper.py --engine fast   --novel <slug> --chapter_num <sel>
uv run python src/scraper.py --engine legacy --novel <slug> --chapter_num <sel>   # failsafe
```
Plus fast-path flags: `--concurrency N`, `--refresh-cookies`, `--delay S`.

### 7.6 Legacy preservation

`--engine legacy` dispatches to today's code path unchanged.

**Decided (owner, 2026-10-25): no automatic fallback.** The fast engine is the
default; legacy runs only when `--engine legacy` is passed explicitly. A fast
path that cannot acquire cookies fails loudly rather than silently opening a
browser window.

### 7.7 Builder calls (settled; flagged so you can veto)

- **Overwrite policy:** fast mode **skips chapters that already exist** and only
  fills gaps/new ones by default, with an explicit `--overwrite` to force. Writes
  go to a temp file and are renamed into place only after a content-plausibility
  check, so a partial/interstitial 200 can never clobber a good file.
- **Failure taxonomy:** a 403 whose body still contains the challenge marker is
  treated as *cookie expiry* (refresh once, single-flight across workers); a 403
  **without** the marker, or repeated timeouts, are treated as *rate-limiting*
  (backoff and reduce concurrency). Exact signals verified during implementation.
- **Novel identification:** fast mode **requires a site slug** (`--novel
  the-beginning-after-the-end`). Name→slug search stays in legacy for now.
- **Genericity:** the fast engine reads `base_url`, chapter/content/title
  selectors from `scraper_config.json`, with a new per-site URL template; it is
  not hardcoded to one site. **Implemented:** the template lives in the
  `fast_chapter_url` key (placeholders `{novel_slug}`/`{slug}`) and the harvest
  seed in `fast_seed_slug`, used by both `chapter_url()` and `harvest_session`.
- **Content bar:** "identical" means **paragraph text equal to a fresh legacy
  scrape**, not byte-equal to the pre-existing corpus (which legacy itself no
  longer reproduces byte-for-byte).

## 8. Open questions for the builder

All owner-facing questions are now settled (§7.4 numbering, §7.6 no auto-fallback,
§7.3 concurrency). Remaining items are builder calls already recorded in §7.7:

1. Numbering/storage scheme — **A** (page-title number).
2. Auto-fallback to legacy — **off**.
3. Cookie cache location — gitignored `.scraper_cache/` (builder call).
4. Parallelism default — **3–4 workers** (§7.3).
5. New optional dependency `curl_cffi` — accepted as a `scraper` extra (builder
   call; confirm at implementation).

## 9. Verification plan

- Byte-compare fast-fetched chapters against existing local files for the same
  chapters (ch1, ch5) — expect identical content.
- Request an unmapped number and confirm it is reported and skipped without
  writing a file or crashing (the site itself skips 510–512; its chapter list
  jumps from 509 to 513, so there is nothing to fetch there).
- Time a 20-chapter run sequentially vs `--concurrency 4`.
- Confirm no window appears on screen and focus is never taken.
- Confirm a stale/blanked cookie triggers exactly one refresh, then continues.

## 10. Rollout

1. Build fast engine + tests for pure helpers (title/section mapping, cookie
   cache I/O, retry logic) — there is no test suite today.
2. Validate on a small range; compare against legacy output.
3. Flip default to fast; README documents both engines and the fallback.
4. (Separate) commit the pending summarizer/env work — unaffected by this work.

## 11. Addendum — harvest is pluggable (2026-10-29)

**What changed on the sites.** The Cloudflare-protected site that motivated §2 went
down and returned at a new domain, and the domain the config pointed at began
serving a JS "Loading…" redirect stub instead of pages. A different host serving the
same novel is reachable over plain HTTP with no bot check at all, and exposes a
paginated chapter index. So the *browser-once-then-HTTP* model is still right, but
"harvest the chapter list" must not be hardwired to a dropdown in a browser.

**Design.** The harvest step — "produce the ordered list of (slug, title)" — is
selected per site by a `harvest` key:

| `harvest` | Implementation | Cost |
| --- | --- | --- |
| `"browser"` (default when the key is absent) | §2's path: off-screen Chrome, pass the check, read `select.select-chapter` and cookies | ~30 s, needs Chrome |
| `"paginated"` | Plain `curl_cffi` walk of the chapter index | ~13 s, no browser |

Both return the same `SessionData`, so **nothing downstream changes**: the cookie
cache, `chapter_map()`, skip-existing, the plausibility guard and the atomic writer
are shared. The browser path is preserved verbatim — when the protected site is
reachable again, switching its entry back to `"browser"` restores the original
behaviour with no code change.

**Pagination style is configuration, not inference.** A `{page}` placeholder in
`chapter_list_page_url` means numbered pages: walk upward until a page contributes no
new slugs. Otherwise `chapter_list_next_selector` is followed as a next-link chain.
Guessing between the two was rejected — a wrong guess terminates the walk on page 1
and looks like a short novel.

**No automatic fallback (consistent with §7.6).** If a paginated site starts
challenging, the run fails with an error naming the site and telling the operator to
set `"harvest": "browser"`. Silently opening a browser would hide a stale config and
pay the 30 s cost invisibly.

### 11.1 Two bugs found while verifying against the live site

- **Challenge false positive.** Challenge detection was the bare substring
  `"just a moment"`. Chapter prose contains that phrase, so a genuine 99 KB chapter
  was classified as an interstitial and rejected. Markers were probed against live
  pages: `challenge-platform` appears on *ordinary* pages too (Cloudflare injects it
  everywhere) and is unusable; `_cf_chl_opt`, `cf-browser-verification` and
  "enable javascript and cookies to continue" appear only on real challenges.
  Detection now keys on those structural markers.
- **Silent list truncation.** The site throttles bursts with `429` + `Retry-After:
  10` after roughly five rapid list pages. The transport returned 429 immediately,
  and the walker read "a page with no new links" as end-of-list — producing a
  complete-looking map of 400 chapters. That is worse than a failure, because later
  runs would skip chapters as "not in the site's list". Now 429 is retried with
  bounded backoff, and a non-200 list page is a hard error.

**Lesson recorded:** in any list-walking code, "no items found" must never be the
same branch as "the list ended". Distinguish "the page was read and was empty"
from "the page could not be read".

### 11.2 Verification evidence (2026-10-29)

- Full paginated harvest: 529 chapters in 13 s, including one automatic 429 backoff.
- Corpus gap chapters 510–512 scraped in 0.9 s, titles and head/tail intact.
- Browser harvest on the protected site still returns 533 options (~32 s), and its
  chapters still fetch and extract — the preserved path remains working.
- Parity spot-check against existing corpus text: `SequenceMatcher` ratio 0.9926,
  0.9959 by body lines; the only difference is two junk trailer lines present in the
  old corpus files.
- Corpus complete afterwards: 516 chapters, 1–516, no gaps.
- Suite: 77 passed.


### 11.3 Corpus naming and site furniture (2026-10-30)

Two corpus-level problems were fixed outside the scraper's request path.

**Five-digit chapter padding.** Chapters were written as `chapter_1.txt` through
`chapter_516.txt`, which sorts lexicographically as 1, 10, 100, 101, ... Files are
now written as `chapter_<n:05d>.txt`. This is a local metadata change: re-scraping
to obtain the new names would re-download everything and risk replacing good local
text with whatever the site serves today, so `corpus_tools rename-padding` renames
in place. The two deprecated forms (`chapter_<n:04d>.txt` from the legacy Selenium
engine and unpadded `chapter_<n>.txt`) remain recognised on *read* so a chapter
that is already on disk under an old name is never re-scraped as a duplicate; the
lookups are a migration window and can be deleted once no corpus uses them.

**Trailer removal is statistical, not string-based.** 513 of 516 chapters ended in
1-2 junk lines of three distinct texts. Naming those texts in code would have been
correct for about a day: a live probe of the current site showed it injects *no*
ad line at all, so the 513 trailers were residue from the previous site. The tool
therefore learns the strings: a line qualifies as site furniture when it ends at
least 10% of the directory's chapters. Real prose does not repeat verbatim across
hundreds of chapters; footers do. Removal only walks in from the end and stops at
the first non-recurring line, so mid-chapter dialogue that quotes the footer
survives, and a file is never emptied.

**Lesson recorded:** when cleaning scraped text, derive what to remove from the
corpus rather than from the site's current DOM. Site furniture is a property of
when the text was scraped, not of the site today.

Reported plan on the real corpus: 516 renames, 848 lines in 513 files, three
distinct trailer texts discovered with no hardcoded strings.

**Verification evidence:** every migrated file parsed byte-identical to its backup
once trailer lines were discounted (0 of 516 differ); all 516 names match
`chapter_\d{5}\.txt`; range is 1-516 with no gaps or duplicates; the one
lookalike real line ("thank you thank you thank you." in chapter 22) survived;
`existing_chapter` resolves all four sampled chapters under the new names; the
summarizer reads 516 contiguous chapters. Suite: 90 passed.
