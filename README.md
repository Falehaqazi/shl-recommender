# SHL Assessment Recommender

A conversational agent that takes a user from a vague hiring intent
("I'm hiring a Java developer") to a grounded shortlist of SHL
assessments through dialogue. Built for the SHL AI Intern take-home.

## Quickstart

```bash
# 1. Install
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your GROQ_API_KEY (free at https://console.groq.com/keys)

# 3. Normalize the provided catalog
python -m scripts.normalize_catalog
# -> writes data/catalog.json (377 items)

# 4. Build the eval set from sample traces
python -m scripts.build_eval_set
# -> writes data/eval_set.json

# 5. Run the API locally
uvicorn app.main:app --reload
# -> http://127.0.0.1:8000/health  -> {"status": "ok"}
# -> POST http://127.0.0.1:8000/chat

# 6. Run the eval harness
python -m scripts.eval_harness --mode offline
```

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /chat`

Stateless: every request carries the full conversation history.

Request:
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

Response:
```json
{
  "reply": "Here's a 5-item battery for a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is `[]` when clarifying or refusing; a 1–10 item array
when committing. `end_of_conversation` is `true` only when the user has
explicitly signaled satisfaction.

## Architecture

```
POST /chat
    │
    ▼
┌─────────────────────────┐
│  Router (LLM)           │  classifies intent into one of:
│                         │  clarify / recommend / refine / compare /
│                         │  soft_decline / refuse
└──────────┬──────────────┘
           │
   ┌───────┼────────┐
   ▼       ▼        ▼
clarify recommend refine ... (specialist nodes)
   │       │        │
   └───────┴────────┘
           │
           ▼
┌─────────────────────────┐
│  build_response         │  guardrail:
│                         │  - drop non-catalog URLs
│                         │  - cap 1..10
│                         │  - set end_of_conversation only on user signal
│                         │  - validate test_type letters
└──────────┬──────────────┘
           │
           ▼
       ChatResponse
```

### Retrieval

Hybrid: BM25 (lexical) + dense (BAAI/bge-small-en-v1.5, cosine) fused
with Reciprocal Rank Fusion (RRF, k=60). Pure dense smears across
near-duplicates ("Java 8 (New)" vs "Java 11"); pure BM25 misses
semantic matches ("leadership test" → OPQ). Hybrid catches both.
RRF is parameter-free.

### Agent

LangGraph state machine with explicit intent routing. Each node has one
job; the router only decides which node runs. This is more debuggable
and testable than a single mega-prompt, and the LangGraph trace gives
us free per-turn debugging during eval.

### Guardrails

`build_response` is the single egress chokepoint. Every shortlist URL
is re-checked against the catalog before being returned. Any LLM
hallucination of a URL is silently dropped. The schema (Pydantic v2)
enforces 1–10 cap, SHL-domain URLs, valid test_type letters at the
serialization layer too — defense in depth.

The previous shortlist is recovered from history via an HTML-comment
sentinel embedded in our own assistant replies. This keeps the API
stateless while letting refine and compare turns reference earlier
recommendations.

## Files

```
app/
├── main.py              FastAPI app, /health + /chat, lifespan loads catalog
├── schemas.py           Pydantic v2 request/response models
├── config.py            Settings (env vars)
├── llm.py               Groq + Gemini fallback, JSON mode
├── agent/
│   ├── graph.py         LangGraph state machine
│   ├── nodes.py         Six specialist node implementations
│   └── prompts.py       All prompts, one file
└── retrieval/
    ├── catalog.py       AssessmentItem + Catalog loader
    └── index.py         Hybrid retriever (BM25 + dense + RRF)

scripts/
├── normalize_catalog.py Convert provided catalog_raw.json -> catalog.json
├── build_eval_set.py    Parse sample traces -> eval_set.json
├── eval_harness.py      Replay traces, compute Recall@10, run probes
└── scrape_catalog.py    Backup scraper (not used; catalog provided)

data/
├── catalog_raw.json     The provided SHL catalog (input)
├── catalog.json         Normalized catalog (377 items)
├── traces/              Sample conversations (markdown)
├── eval_set.json        Parsed traces with gold shortlists
└── eval_report.json     Latest eval run output

tests/
└── test_schema.py       Hard-eval schema guards (pytest)
```

## Deployment

Render free tier. `render.yaml` is committed; pushing to a connected
GitHub repo deploys automatically. Set `GROQ_API_KEY` (and optionally
`GEMINI_API_KEY`) in the Render dashboard.

Cold start is ~30–60s (within SHL's 2-minute /health budget). After
warm-up, /chat typically responds in 2–4 seconds.

## Evaluation results

(filled in after running `scripts/eval_harness.py` — see `data/eval_report.json`)

## Design choices (interview defense, summary)

| Choice                            | Why                                        | Tradeoff                                  |
|-----------------------------------|--------------------------------------------|-------------------------------------------|
| Hybrid retrieval (BM25+dense+RRF) | Catalog has high lexical overlap; dense alone smears, BM25 alone misses semantic matches | More moving parts than pure dense |
| Groq Llama-3.3-70B primary        | ~500 tok/s throughput leaves room for 3 LLM calls/turn in 30s budget | Less reasoning power than GPT-4-class |
| Gemini Flash fallback             | Survives Groq rate limits during eval     | Two providers to keep up to date |
| LangGraph multi-node              | Each intent has its own prompt and unit test | Slightly more boilerplate than flat function |
| Sentinel-in-reply for state       | Keeps API stateless while preserving refine/compare context | Sentinel must not leak into evaluator's reply parsing — we use HTML comments which most renderers hide |
| Prose reply, no markdown table    | Evaluator reads structured recommendations; tables burn tokens and add hallucination surface | Slightly less informative for human readers |
| Guardrail re-checks every URL     | Defense in depth against LLM hallucination | Tiny CPU cost on every response           |

## What didn't work / what I'd do next

- Initial draft used a single mega-prompt for the whole agent. It
  worked on the happy path but couldn't distinguish refine from
  recommend without a dedicated router.
- Pure dense retrieval scored well on conceptual queries (C1
  leadership) but poorly on technical-stack queries (C2 Rust, C9 Java
  full-stack) where exact tokens matter. Adding BM25 fixed both.
- Future: a re-ranker (cohere-rerank or a cross-encoder) over the
  RRF top-30 would likely improve Recall@10 by another 5-10 points on
  the harder traces. Skipped for time.

## AI assistance used

Claude (claude.ai) for: scaffold generation, prompt iteration against the provided traces, trace pattern extraction, eval harness code. All design decisions — hybrid retrieval over pure dense, six-intent router over a mega-prompt, HTML-comment sentinel for stateless state recovery, separate soft-decline vs hard-refuse paths — were mine, made by reading the 10 provided traces and observing failure modes. See DESIGN_LOG.md for the trace-driven reasoning.
