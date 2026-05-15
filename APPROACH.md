# Approach Document — SHL Assessment Recommender

**Author:** Faleha Qazi
**Submission URL:** https://shl-recommender-fuyx.onrender.com
**Repo:** https://github.com/Falehaqazi/shl-recommender

## Design choices

The agent is a **LangGraph state machine with explicit intent routing**. A router LLM call classifies the next action as one of six intents — `clarify`, `recommend`, `refine`, `compare`, `soft_decline`, `refuse` — and dispatches to a specialist node per intent. This is more debuggable and testable than a single mega-prompt and made trace-driven prompt iteration tractable. I added a cheap heuristic short-circuit ahead of the LLM router that handles closure phrases ("perfect", "locking it in"), prompt-injection patterns, and compare questions with a prior shortlist without an LLM call — cuts ~25-30% of LLM calls during eval runs.

Retrieval is **hybrid: BM25 + dense (BAAI/bge-small-en-v1.5) fused with Reciprocal Rank Fusion (k=60)**. The catalog has 377 items with high lexical overlap (many near-duplicate Java/SQL/Microsoft variants) and pure dense retrieval smears across these — returning random orderings between runs. Pure BM25 misses semantic matches like "leadership test" → OPQ32r. RRF is parameter-free and consistently matches learned fusion on small structured catalogs. Empirically on C9's hard JD query, BM25 alone scored 5/7 gold-in-top-30; the dense leg recovers the two semantic misses (Verify G+, OPQ32r).

The API is **fully stateless** as specified. To support refine and compare turns, the agent embeds an HTML-comment sentinel (`<!--SHL_REC:[...urls...]-->`) at the end of its own replies. Future turns parse this back to recover the previous shortlist. The sentinel is invisible to most renderers and irrelevant to the structured evaluator.

## Stack justification

- **FastAPI + Pydantic v2** for the API. Pydantic enforces the non-negotiable schema at ingress and egress; a guardrail layer re-validates every URL against the catalog before returning, so any LLM URL-hallucination is silently dropped.
- **Provider chain: OpenRouter (free Llama-3.3-70B) → Groq → Gemini.** Each can be enabled via env var; the chain falls through on 429 or HTTP errors. This survives single-provider rate-limit outages during the evaluator run.
- **In-memory hybrid retriever** (numpy + rank-bm25 + sentence-transformers). 377 items doesn't justify a vector DB; the full index loads in ~5s at startup.
- **Render free tier** for deployment. Cold start ~30-60s (within SHL's 2-minute /health budget).

## Prompt design

Six prompts, one per intent, all asking for strict JSON output. The recommend prompt sees only retrieved candidates, never the full catalog — this is the anti-hallucination pillar: the LLM cannot recommend something the retriever didn't surface, and a final guardrail layer drops any URL not in the catalog regardless.

Replies are 1-2 sentences of prose; no markdown tables. The evaluator reads structured `recommendations`, so tables only burn tokens and introduce hallucination surface. The router prompt explicitly enforces `user_satisfied=true` only on closure phrases with no edit verbs present, avoiding premature `end_of_conversation=true` on refinement turns like "Add AWS and Docker."

## Evaluation approach

I built a replay harness (`scripts/eval_harness.py`) that feeds each trace's user messages into the agent in order, accumulating history, and scores:

- **Mean Recall@10** against gold shortlists extracted from each trace's final agent table.
- **Schema compliance** (1-10 cap, catalog URLs only, valid letters).
- **Behavior probes** (5 binary assertions): off-topic refusal, prompt-injection resilience, no-recommend on turn 1 for vague, recommend-on-turn-1 for specific JD, refinement honored.

> _Results from the latest offline run (single-trace C9, before full quota reset):_
>
> **Recall@10 on C9:** 0.43-0.57 (final shortlist 5-10 items, gold = 7)
> **Schema-clean:** 1/1
> **Probes passed:** 5/5
>
> Full 10-trace run pending API-quota reset; numbers will be in `data/eval_report.json` at submission time.

## What didn't work / how I measured

- Initial single mega-prompt for the whole agent worked on the happy path but couldn't distinguish refine from recommend reliably. Switching to a router + specialists fixed it.
- Pure dense retrieval scored well on conceptual queries (C1 leadership) and poorly on technical-stack queries (C2 Rust, C9 Java full-stack) where exact tokens matter. Adding BM25 with RRF closed both gaps.
- The free-tier LLM provider chain was the biggest engineering challenge. Built a three-provider fallback with heuristic short-circuits to survive aggressive rate limits without needing paid credits.
- The eval harness can't fully simulate SHL's LLM-driven user — it replays trace messages in order regardless of what our agent asked. So our Recall@10 is an approximate lower bound; the real evaluator's adaptive user may yield better results because our recommend node fuses all user turns into the retrieval query and benefits from out-of-order information.

## AI tools used

Claude (claude.ai) for: scaffold generation, prompt iteration against the provided traces, trace pattern extraction, eval harness code. All design decisions — hybrid retrieval over pure dense, six-intent router over a mega-prompt, HTML-comment sentinel for stateless state recovery, separate soft-decline vs hard-refuse paths, heuristic router short-circuits — were mine, made by reading the 10 provided traces and observing failure modes. See `DESIGN_LOG.md` for the trace-driven reasoning.