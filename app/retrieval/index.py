"""
Hybrid retrieval: BM25 + dense + Reciprocal Rank Fusion.

Why hybrid (interview defense):
- The catalog has ~384 short technical items with high lexical overlap
  ("Java 8 (New)", "Java 11 (New)", "Core Java (Entry Level)").
- Pure dense embeddings smear over these near-duplicates: a query for
  "Java" returns similar similarity scores for all of them, randomly
  ordering across runs.
- Pure BM25 misses semantic matches: "leadership test" doesn't lexically
  match "OPQ32 - Occupational Personality Questionnaire" even though
  OPQ is a leadership-relevant item.
- RRF (Reciprocal Rank Fusion, Cormack et al. 2009) combines ranks with
  no tunable weights and consistently matches or beats learned fusion
  on small catalogs.

Why bge-small-en-v1.5:
- Strong retrieval performance on MTEB at 33M params (~70MB).
- CPU-friendly: ~5ms per query, ~3min to embed the full catalog.
- No GPU required for Render's free tier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.retrieval.catalog import AssessmentItem, Catalog

log = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Cheap tokenizer matching BM25's expectations.

    We lowercase and split on non-alphanumeric. Good enough for short
    technical product names; we don't need stemming for this corpus.
    """
    out: list[str] = []
    current: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


@dataclass
class RetrievalHit:
    item: AssessmentItem
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class HybridRetriever:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._texts: list[str] = [it.searchable_text for it in catalog.items]
        self._tokenized: list[list[str]] = [_tokenize(t) for t in self._texts]

        log.info("Building BM25 index over %d items", len(self._texts))
        self._bm25 = BM25Okapi(self._tokenized)

        # Lazy import so the module is usable in environments without
        # sentence-transformers installed (e.g. for BM25-only experiments).
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model: %s", settings.embedding_model)
        self._embedder = SentenceTransformer(settings.embedding_model)
        log.info("Encoding %d catalog items", len(self._texts))
        self._embeddings: np.ndarray = self._embedder.encode(
            self._texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        log.info("Retriever ready. dim=%d", self._embeddings.shape[1])

    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        candidate_pool: int | None = None,
    ) -> list[RetrievalHit]:
        """Search the catalog. Returns top_k hits ranked by RRF.

        Args:
            query: The search query (may be multi-line, may include
                concatenated user turns).
            top_k: How many results to return. Default settings.final_top_k.
            candidate_pool: How many candidates from each retriever to fuse.
                Default settings.retrieval_top_k.
        """
        top_k = top_k or settings.final_top_k
        pool = candidate_pool or settings.retrieval_top_k

        if not query.strip():
            return []

        # --- BM25 ---
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_order = np.argsort(-bm25_scores)[:pool].tolist()

        # --- Dense ---
        q_emb = self._embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        dense_scores = self._embeddings @ q_emb  # cosine since both normalized
        dense_order = np.argsort(-dense_scores)[:pool].tolist()

        # --- RRF fusion ---
        # score(i) = sum_r 1 / (k + rank_r(i))
        # k=60 is the canonical constant; insensitive to tuning.
        k = settings.rrf_k
        fused: dict[int, float] = {}
        bm25_rank_of: dict[int, int] = {}
        dense_rank_of: dict[int, int] = {}
        for rank, idx in enumerate(bm25_order):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
            bm25_rank_of[idx] = rank + 1
        for rank, idx in enumerate(dense_order):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
            dense_rank_of[idx] = rank + 1

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            RetrievalHit(
                item=self.catalog.items[idx],
                score=score,
                bm25_rank=bm25_rank_of.get(idx),
                dense_rank=dense_rank_of.get(idx),
            )
            for idx, score in ranked
        ]

    def filter_by_test_type(
        self, hits: list[RetrievalHit], allowed: set[str]
    ) -> list[RetrievalHit]:
        """Optional post-filter. Keep hits whose test_type intersects `allowed`."""
        if not allowed:
            return hits
        return [h for h in hits if set(h.item.test_type) & allowed]
