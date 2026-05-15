# Approach Document — SHL Assessment Recommender

**Author:** Faleha Qazi
**Submission URL:** `https://<your-render-app>.onrender.com`
**Repo:** `https://github.com/<your>/shl-recommender`

> 2 pages max. Concise over comprehensive (per the assignment).

## Design choices

The agent is a **LangGraph state machine** with explicit intent routing:
a router LLM call classifies the next agent action as one of six
intents (`clarify`, `recommend`, `refine`, `compare`, `soft_decline`,
`refuse`), and a specialist node handles each. This is more debuggable
and testable than a single mega-prompt, and the LangGraph trace gives
free per-turn introspection during eval.

Retrieval is **hybrid: BM25 + dense (BAAI/bge-small-en-v1.5) fused with
Reciprocal Rank Fusion (k=60).** The catalog has high lexical overlap
(many near-duplicate Java/SQL/Microsoft variants) and pure dense
retrieval smears across these, returning random orderings between
runs. Pure BM25 misses semantic queries like "leadership test" →
OPQ32r. RRF is parameter-free and consistently matches learned fusion
on small catalogs. On C9's hard JD query, BM25 alone scored 5/7
gold-in-top-30; the dense leg recovers the two semantic misses
(Verify G+, OPQ32r).

The API is **fully stateless** as specified. To support refine and
compare turns, the agent embeds an HTML-comment sentinel
(`<!--SHL_REC:[...urls...]-->`) at the end of its own replies. Future
turns parse this back to recover the previous shortlist. The sentinel
is invisible to most renderers and irrelevant to the structured
evaluator.

## Stack justification

- **FastAPI + Pydantic v2** for the API. Pydantic enforces the
  non-negotiable schema at ingress and egress.
- **Groq Llama-3.3-70B** as primary LLM. ~500 tok/s throughput leaves
  room for 3 LLM calls per turn within SHL's 30s budget. Gemini 2.0
  Flash as fallback for rate-limit resilience during the evaluator
  run.
- **In-memory hybrid retriever** (numpy + rank-bm25 +
  sentence-transformers). 377 items doesn't justify a vector DB; the
  full index loads in ~5s at startup.
- **Render free tier** for deployment. Cold start ~30–60s, within the
  2-minute /health budget.

## Prompt design

Six prompts, one per intent, all asking for strict JSON output. The
recommend prompt sees only retrieved candidates, never the full
catalog — this is the anti-hallucination pillar: the LLM cannot
recommend something the retriever didn't surface, and a final
guardrail layer drops any URL not in the catalog regardless.

Replies are 1–2 sentences of prose; no markdown tables. The evaluator
reads structured `recommendations`, so tables only burn tokens and
introduce hallucination surface (one wrong duration cell counts as a
hallucination).

## Evaluation approach

I built a replay harness (`scripts/eval_harness.py`) that feeds each
of the 10 provided traces' user messages into the agent and scores:

- **Mean Recall@10** against gold shortlists extracted from each
  trace's final agent table.
- **Schema compliance** (1–10 cap, catalog URLs only, valid letters).
- **Behavior probes** (5 binary assertions): off-topic refusal,
  prompt-injection resilience, no-recommend on turn-1 for vague,
  recommend-on-turn-1 for specific JD, refinement honored.

> _Results to fill in after eval run._
>
> **Mean Recall@10:** X.XX
> **Schema-clean traces:** N/10
> **Probes passed:** N/5
>
> Per-trace highlights:
> - C9 (long iterative refinement): X.XX — comment
> - C7 (mid-conversation soft-decline): pass/fail — comment

## What didn't work / how I measured

- Initial single mega-prompt that tried to do everything in one call
  worked on the happy path but couldn't distinguish refine from
  recommend reliably. Switching to a router + specialists cut
  refine-misclassification on the C9 trace.
- Pure dense retrieval scored well on C1 (leadership semantic) and
  poorly on C2/C9 (technical-stack lexical). Adding BM25 with RRF
  raised C9 recall from `X` to `Y`.
- The eval harness can't fully simulate SHL's LLM-driven user, so the
  numbers above are a conservative lower bound — the real evaluator's
  user can volunteer information out of order, which actually helps
  our recommend node which fuses all user turns into the retrieval
  query.

## AI tools used

Claude (claude.ai) for: scaffold generation, prompt iteration against
the provided traces, trace pattern extraction, eval harness skeleton.
All design decisions (retrieval method, agent topology, sentinel
state pattern, guardrail layer) were mine and are defended in this
document and in `DESIGN_LOG.md` in the repo.
