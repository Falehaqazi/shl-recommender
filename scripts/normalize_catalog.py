"""
Convert the provided SHL catalog (data/catalog_raw.json) into the clean
schema the app expects (data/catalog.json).

What the raw file looks like:
- 377 items
- Field names: name, link, keys, job_levels, languages, duration, remote, adaptive, description
- `keys` holds full test-type names like "Knowledge & Skills"
- `remote`/`adaptive` are "yes"/"no" strings
- `duration` is a string like "30 minutes" (or "")
- The raw JSON has unescaped newlines inside the "Microsoft 365" name string,
  so vanilla json.load() fails. We patch with a regex pre-pass.

What we output:
- A list of dicts with: name, url, test_type (list of single letters),
  description, job_levels (list), languages (list), duration_minutes (int|None),
  remote_testing (bool), adaptive (bool).

Usage:
    python -m scripts.normalize_catalog
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RAW_PATH = Path("data/catalog_raw.json")
CLEAN_PATH = Path("data/catalog.json")

# Map full names (as seen in raw "keys") -> single-letter codes (as seen in
# the index page and the spec example response).
# CAREFUL: their data spells it "Biodata & Situational Judgment" (singular).
# The catalog legend uses "Judgement" (British). We accept both.
NAME_TO_LETTER: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Personality & Behaviour": "P",  # tolerate British spelling
    "Simulations": "S",
}


def load_raw_lenient(path: Path) -> list[dict]:
    """Load the provided catalog despite its unescaped newlines.

    Strategy: regex-sub every JSON string token, replacing internal \\n / \\r
    with spaces. We match strings as quote, body (escapes allowed), quote.
    """
    text = path.read_text(encoding="utf-8")

    def fix_string(m: re.Match[str]) -> str:
        s = m.group(0)
        return s.replace("\n", " ").replace("\r", " ")

    # The pattern matches: " followed by any (escaped or non-quote) chars, then "
    pattern = re.compile(r'"(?:[^"\\]|\\.)*"', flags=re.DOTALL)
    fixed = pattern.sub(fix_string, text)
    data = json.loads(fixed)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list at root, got {type(data).__name__}")
    return data


def parse_duration(s: str | None) -> int | None:
    """'30 minutes' -> 30. '' -> None. None -> None."""
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def normalize(raw_item: dict) -> dict | None:
    """Map one raw item to our clean schema. Returns None to skip."""
    name = (raw_item.get("name") or "").strip()
    # Multiple spaces from our newline-fix can creep in — collapse them.
    name = re.sub(r"\s+", " ", name)
    url = (raw_item.get("link") or "").strip()
    if not name or not url:
        return None

    # Test type letters
    raw_keys = raw_item.get("keys") or []
    if not isinstance(raw_keys, list):
        raw_keys = [raw_keys]
    letters: list[str] = []
    unmapped: list[str] = []
    for k in raw_keys:
        k_clean = re.sub(r"\s+", " ", str(k).strip())
        letter = NAME_TO_LETTER.get(k_clean)
        if letter:
            if letter not in letters:
                letters.append(letter)
        else:
            unmapped.append(k_clean)
    if unmapped:
        log.warning("  unmapped test-type names for %r: %s", name, unmapped)
    letters.sort()  # stable, alphabetical

    out = {
        "name": name,
        "url": url,
        "test_type": letters,
        "description": re.sub(r"\s+", " ", (raw_item.get("description") or "").strip()),
        "job_levels": [
            re.sub(r"\s+", " ", str(j).strip().rstrip(",")).strip()
            for j in (raw_item.get("job_levels") or [])
            if str(j).strip().rstrip(",").strip()
        ],
        "languages": [
            re.sub(r"\s+", " ", str(l).strip().rstrip(",")).strip()
            for l in (raw_item.get("languages") or [])
            if str(l).strip().rstrip(",").strip()
        ],
        "duration_minutes": parse_duration(raw_item.get("duration") or raw_item.get("duration_raw")),
        "remote_testing": str(raw_item.get("remote", "")).strip().lower() == "yes",
        "adaptive": str(raw_item.get("adaptive", "")).strip().lower() == "yes",
    }
    return out


def main() -> int:
    if not RAW_PATH.exists():
        log.error("Missing %s", RAW_PATH)
        return 1

    log.info("Loading %s", RAW_PATH)
    raw = load_raw_lenient(RAW_PATH)
    log.info("Raw items: %d", len(raw))

    clean: list[dict] = []
    seen_urls: set[str] = set()
    skipped = 0
    for item in raw:
        norm = normalize(item)
        if norm is None:
            skipped += 1
            continue
        if norm["url"] in seen_urls:
            log.warning("Duplicate URL, skipping: %s", norm["url"])
            skipped += 1
            continue
        seen_urls.add(norm["url"])
        clean.append(norm)

    log.info("Clean items: %d (skipped %d)", len(clean), skipped)

    # Coverage stats — useful to print during eval debugging.
    with_desc = sum(1 for c in clean if c["description"])
    with_duration = sum(1 for c in clean if c["duration_minutes"])
    no_test_type = sum(1 for c in clean if not c["test_type"])
    log.info("  with description: %d", with_desc)
    log.info("  with duration:    %d", with_duration)
    log.info("  WITHOUT test_type: %d", no_test_type)

    CLEAN_PATH.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", CLEAN_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
