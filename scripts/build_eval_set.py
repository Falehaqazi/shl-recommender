"""
Parse the 10 sample conversation traces into a structured eval set.

For each trace we extract:
- The user-side turn sequence (what the simulated user "would say").
- The final labeled shortlist (URLs from the last agent table).
- The "expected" intent on each turn — useful for debugging where the
  router goes wrong.

Usage:
    python -m scripts.build_eval_set

Output:
    data/eval_set.json  -- list of dicts ready for scripts/eval_harness.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

TRACES_DIR = Path("data/traces/GenAI_SampleConversations")
OUT_PATH = Path("data/eval_set.json")


URL_RE = re.compile(r"https://www\.shl\.com/products/product-catalog/view/[a-z0-9\-]+/?")
USER_BLOCK_RE = re.compile(
    r"\*\*User\*\*\s*\n+(>\s*[^\n]+(?:\n>\s*[^\n]*)*)",
    re.MULTILINE,
)
TABLE_RE = re.compile(
    r"(\|[^\n]+\|\n\|[\s\-:|]+\|\n(?:\|[^\n]+\|\n)+)"
)
TURN_RE = re.compile(r"^###\s+Turn\s+(\d+)", re.MULTILINE)


def parse_user_blocks(text: str) -> list[str]:
    """Pull each user blockquote out, preserving multi-line content (e.g. JDs)."""
    blocks = []
    for m in USER_BLOCK_RE.finditer(text):
        raw = m.group(1)
        # Each line starts with "> " — strip that prefix.
        lines = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if ln.startswith(">"):
                lines.append(ln[1:].strip())
            elif ln:
                lines.append(ln)
        msg = "\n".join(l for l in lines if l).strip()
        if msg:
            blocks.append(msg)
    return blocks


def parse_trace(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    name = path.stem

    user_msgs = parse_user_blocks(text)

    # Final shortlist: last markdown table's URLs (dedup, preserve order)
    tables = TABLE_RE.findall(text)
    gold_urls: list[str] = []
    if tables:
        seen: set[str] = set()
        for u in URL_RE.findall(tables[-1]):
            u = u.rstrip("/") + "/"  # normalize trailing slash
            if u not in seen:
                gold_urls.append(u)
                seen.add(u)

    # Turn count (highest "### Turn N" in the file)
    turn_nums = [int(m.group(1)) for m in TURN_RE.finditer(text)]
    expected_turns = max(turn_nums) if turn_nums else len(user_msgs)

    return {
        "trace_id": name,
        "user_messages": user_msgs,
        "expected_turns": expected_turns,
        "gold_shortlist": gold_urls,
    }


def main() -> int:
    if not TRACES_DIR.exists():
        log.error("Traces directory not found: %s", TRACES_DIR)
        return 1

    traces = []
    for f in sorted(TRACES_DIR.glob("*.md")):
        t = parse_trace(f)
        log.info(
            "%s: %d user msgs, %d gold items, %d expected turns",
            t["trace_id"], len(t["user_messages"]), len(t["gold_shortlist"]), t["expected_turns"],
        )
        traces.append(t)

    OUT_PATH.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s (%d traces)", OUT_PATH, len(traces))
    return 0


if __name__ == "__main__":
    sys.exit(main())
