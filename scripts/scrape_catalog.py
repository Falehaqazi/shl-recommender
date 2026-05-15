"""
Scrape the SHL Individual Test Solutions catalog.

Usage:
    python scripts/scrape_catalog.py

Outputs:
    data/catalog.json  -- list of dicts ready for app/retrieval/catalog.py

Design notes:
- Two-phase scrape: index pages (~32) for URL+flags+test_type codes,
  then product pages (~384) for descriptions, job levels, languages, duration.
- We ONLY scrape type=1 (Individual Test Solutions). Pre-packaged Job
  Solutions are explicitly out of scope per the assignment.
- Polite 1s delay between requests. Total wall time ~7 min.
- Idempotent: writes intermediate JSON after the index phase so a crash
  during product-page scraping doesn't lose all progress.
- Defensive parsing: every product page has slightly different markup
  for missing fields, so we use lenient extractors that return defaults
  rather than crash.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BASE = "https://www.shl.com"
INDEX_URL = "https://www.shl.com/products/product-catalog/"
DELAY_SECONDS = 1.0
TIMEOUT = 30.0
USER_AGENT = "shl-recommender-research/0.1 (contact: applicant)"

DATA_DIR = Path("data")
INDEX_OUT = DATA_DIR / "catalog_index.json"
FULL_OUT = DATA_DIR / "catalog.json"


def fetch(client: httpx.Client, url: str) -> str:
    """GET with retry-once on transient errors."""
    for attempt in (1, 2):
        try:
            r = client.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt == 2:
                log.error("Failed twice on %s: %s", url, e)
                raise
            log.warning("Retry %s after %s", url, e)
            time.sleep(2.0)
    raise RuntimeError("unreachable")


# ----------------------------------------------------------------------
# Phase 1: Index pages
# ----------------------------------------------------------------------


def parse_index_page(html: str) -> list[dict[str, Any]]:
    """Extract rows from the Individual Test Solutions table on one index page.

    The catalog page has TWO tables: type=2 (Job Solutions) and type=1
    (Individual Test Solutions). We've requested with ?type=1 in the URL,
    but the page may still render both — we filter by the header text.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []

    for table in soup.find_all("table"):
        header_text = " ".join(th.get_text(" ", strip=True) for th in table.find_all("th"))
        if "Individual Test Solutions" not in header_text:
            continue

        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 4:
                continue
            link = cells[0].find("a")
            if not link or not link.get("href"):
                continue

            name = link.get_text(strip=True)
            url = link["href"]
            if url.startswith("/"):
                url = BASE + url

            # Remote testing / Adaptive cells: the SHL page uses a green
            # circle span when "yes". Presence of any non-whitespace child
            # element signals true.
            def cell_flag(td) -> bool:
                return bool(td.find(True)) or bool(td.get_text(strip=True))

            remote = cell_flag(cells[1])
            adaptive = cell_flag(cells[2])
            type_codes = re.findall(r"\b[ABCDEKPS]\b", cells[3].get_text(" ", strip=True))

            rows.append(
                {
                    "name": name,
                    "url": url,
                    "remote_testing": remote,
                    "adaptive": adaptive,
                    "test_type": sorted(set(type_codes)),
                }
            )
    return rows


def discover_max_start(html: str) -> int:
    """Find the largest ?start= value in pagination links for type=1."""
    soup = BeautifulSoup(html, "html.parser")
    starts = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"start=(\d+)&type=1", a["href"])
        if m:
            starts.append(int(m.group(1)))
    return max(starts) if starts else 0


def scrape_index(client: httpx.Client) -> list[dict[str, Any]]:
    """Walk every index page and accumulate row dicts."""
    log.info("Phase 1: scraping index pages")
    first = fetch(client, f"{INDEX_URL}?start=0&type=1")
    max_start = discover_max_start(first)
    log.info("Pagination max start=%d (so %d pages)", max_start, max_start // 12 + 1)

    all_rows: list[dict[str, Any]] = parse_index_page(first)
    seen_urls = {r["url"] for r in all_rows}

    for start in range(12, max_start + 1, 12):
        time.sleep(DELAY_SECONDS)
        url = f"{INDEX_URL}?start={start}&type=1"
        html = fetch(client, url)
        rows = parse_index_page(html)
        added = 0
        for r in rows:
            if r["url"] not in seen_urls:
                all_rows.append(r)
                seen_urls.add(r["url"])
                added += 1
        log.info("start=%d: %d rows (%d new). total=%d", start, len(rows), added, len(all_rows))

    log.info("Phase 1 done: %d unique items", len(all_rows))
    return all_rows


# ----------------------------------------------------------------------
# Phase 2: Product detail pages
# ----------------------------------------------------------------------


def parse_product_page(html: str) -> dict[str, Any]:
    """Extract description, job levels, languages, duration from a product page."""
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, Any] = {
        "description": "",
        "job_levels": [],
        "languages": [],
        "duration_minutes": None,
    }

    # The product page renders #### Section headings followed by paragraphs.
    # We walk h4-ish nodes and grab the next text block.
    for header in soup.find_all(["h4", "h2", "h3"]):
        title = header.get_text(strip=True).lower()
        sibling = header.find_next_sibling()
        if sibling is None:
            continue
        body = sibling.get_text(" ", strip=True)
        if not body:
            continue

        if "description" in title:
            out["description"] = body
        elif "job level" in title:
            out["job_levels"] = [
                s.strip().rstrip(",")
                for s in body.split(",")
                if s.strip().rstrip(",")
            ]
        elif "language" in title:
            out["languages"] = [
                s.strip().rstrip(",")
                for s in body.split(",")
                if s.strip().rstrip(",")
            ]
        elif "length" in title or "completion time" in title:
            m = re.search(r"(\d+)", body)
            if m:
                out["duration_minutes"] = int(m.group(1))

    # Fallback: scan the whole page for "Approximate Completion Time in minutes = N"
    if out["duration_minutes"] is None:
        m = re.search(r"completion time in minutes\s*=\s*(\d+)", html, re.I)
        if m:
            out["duration_minutes"] = int(m.group(1))

    return out


def enrich_with_details(
    client: httpx.Client, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    log.info("Phase 2: enriching %d items with product-page details", len(rows))
    enriched: list[dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        time.sleep(DELAY_SECONDS)
        try:
            html = fetch(client, row["url"])
            details = parse_product_page(html)
            enriched.append({**row, **details})
        except Exception as e:
            log.warning("Failed on %s: %s. Keeping index-only data.", row["url"], e)
            enriched.append({**row, "description": "", "job_levels": [], "languages": [], "duration_minutes": None})

        if i % 25 == 0:
            log.info("Enriched %d/%d", i, len(rows))
            # Incremental save in case of crash.
            FULL_OUT.write_text(json.dumps(enriched, indent=2), encoding="utf-8")

    return enriched


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        index_rows = scrape_index(client)
        INDEX_OUT.write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
        log.info("Wrote %s", INDEX_OUT)

        full = enrich_with_details(client, index_rows)
        FULL_OUT.write_text(json.dumps(full, indent=2), encoding="utf-8")
        log.info("Wrote %s (%d items)", FULL_OUT, len(full))

    return 0


if __name__ == "__main__":
    sys.exit(main())
