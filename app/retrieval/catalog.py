"""
Catalog loading and the in-memory data model.

Design notes:
- One AssessmentItem per catalog entry. Stored as a frozen dataclass so
  it's hashable and accidentally-immutable.
- We normalize the test_type field to a sorted tuple of single letters,
  matching the SHL legend. This makes filtering by test-type trivial.
- We build a search blob (`searchable_text`) at load time concatenating
  name + description + job levels + test type names. BM25 indexes this
  blob; dense embeddings encode this blob. One blob, two indexes.
- We index by URL because that's the field the evaluator checks against
  the catalog. Two items with the same URL = data bug, fail loudly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# Map test-type code -> human-readable name. Used in searchable_text
# so a user query like "personality test" can match items tagged "P".
TEST_TYPE_NAMES: dict[str, str] = {
    "A": "Ability and Aptitude",
    "B": "Biodata and Situational Judgement",
    "C": "Competencies",
    "D": "Development and 360",
    "E": "Assessment Exercises",
    "K": "Knowledge and Skills",
    "P": "Personality and Behavior",
    "S": "Simulations",
}


@dataclass(frozen=True)
class AssessmentItem:
    name: str
    url: str
    test_type: tuple[str, ...]  # e.g. ("A", "B", "P") — sorted single letters
    description: str = ""
    job_levels: tuple[str, ...] = field(default_factory=tuple)
    languages: tuple[str, ...] = field(default_factory=tuple)
    duration_minutes: int | None = None
    remote_testing: bool = False
    adaptive: bool = False

    @property
    def test_type_str(self) -> str:
        """Comma-joined for the API response, e.g. 'A,B,P'.

        Matches the format used in SHL's provided conversation traces
        (e.g. "K,S" for an item tagged both Knowledge and Simulations).
        """
        return ",".join(self.test_type)

    @property
    def searchable_text(self) -> str:
        """The blob both BM25 and dense embeddings see."""
        type_names = ", ".join(TEST_TYPE_NAMES.get(t, t) for t in self.test_type)
        parts = [
            self.name,
            self.description,
            f"Test types: {type_names}",
            f"Job levels: {', '.join(self.job_levels)}" if self.job_levels else "",
            f"Duration: {self.duration_minutes} minutes" if self.duration_minutes else "",
            "Remote testing available" if self.remote_testing else "",
            "Adaptive test" if self.adaptive else "",
        ]
        return " | ".join(p for p in parts if p)


class Catalog:
    """In-memory catalog with lookup helpers."""

    def __init__(self, items: list[AssessmentItem]) -> None:
        self.items: list[AssessmentItem] = items
        self._by_url: dict[str, AssessmentItem] = {it.url: it for it in items}
        self._by_name_lower: dict[str, AssessmentItem] = {
            it.name.lower(): it for it in items
        }
        if len(self._by_url) != len(items):
            log.warning(
                "Duplicate URLs in catalog: %d items, %d unique URLs",
                len(items),
                len(self._by_url),
            )

    def __len__(self) -> int:
        return len(self.items)

    def has_url(self, url: str) -> bool:
        return url in self._by_url

    def by_url(self, url: str) -> AssessmentItem | None:
        return self._by_url.get(url)

    def by_name(self, name: str) -> AssessmentItem | None:
        """Loose name lookup. Useful for 'compare OPQ and GSA' style queries."""
        return self._by_name_lower.get(name.lower().strip())


def load_catalog(path: str | Path) -> Catalog:
    """Load catalog JSON from disk. Expects a list of dicts.

    Each dict must have at least: name, url, test_type (list[str]).
    Other fields default if missing.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog file not found at {path}. "
            "Run scripts/scrape_catalog.py or place the SHL-provided file there."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[AssessmentItem] = []
    for r in raw:
        items.append(
            AssessmentItem(
                name=r["name"].strip(),
                url=r["url"].strip(),
                test_type=tuple(sorted(set(r.get("test_type", [])))),
                description=(r.get("description") or "").strip(),
                job_levels=tuple(r.get("job_levels", [])),
                languages=tuple(r.get("languages", [])),
                duration_minutes=r.get("duration_minutes"),
                remote_testing=bool(r.get("remote_testing", False)),
                adaptive=bool(r.get("adaptive", False)),
            )
        )
    log.info("Loaded %d catalog items from %s", len(items), path)
    return Catalog(items)
