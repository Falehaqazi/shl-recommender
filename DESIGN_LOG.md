# Design Log — lessons from inspecting provided files

## Catalog (data/catalog.json, normalized from data/catalog_raw.json)

- **377 items**, all Individual Test Solutions (the "type=1" subset).
- All items are `remote_testing=true`. Useless as a filter; do not pretend it differentiates anything.
- 37 items are adaptive; everything else is non-adaptive. Useful filter.
- 291/377 (77%) have a duration in minutes. The rest will show "—" in reply tables.
- Test-type letter distribution:
  - K (Knowledge & Skills): 240
  - P (Personality & Behavior): 66
  - S (Simulations): 43
  - A (Ability & Aptitude): 32
  - C (Competencies): 19
  - B (Biodata & SJT): 17
  - D (Development & 360): 7
  - E (Assessment Exercises): 2
- Raw file has unescaped newlines inside one name string ("Microsoft Excel 365 - Essentials (New)"). `scripts/normalize_catalog.py` handles this with a regex pre-pass.
- Raw "keys" spelling is "Biodata & Situational Judgment" (singular) — different from the catalog legend's "Judgement". Both mapped to "B".

## API response format clarifications from traces

- `test_type` in a Recommendation is **comma-joined letters with no space**: `"K"`, `"K,S"`, `"A,P,B"`.
- The human-readable `reply` field in traces re-states the shortlist as a markdown table with columns: `# | Name | Test Type | Keys | Duration | Languages | URL`. The evaluator only reads structured `recommendations`, but matching the trace style helps any manual scorer skim.
- When the agent refuses or clarifies, traces show `recommendations: null` (the markdown says "_No recommendations this turn_"). Our schema uses `[]` per the spec — both should be equivalent for the evaluator since it checks for empty.

## Conversational behavior patterns (10 traces)

### Trigger for clarify vs recommend

- "We need a solution for senior leadership" (C1 T1) → **clarify**. Even seniority alone isn't enough; the agent asks who it's for.
- "I'm hiring a senior Rust engineer for high-performance networking" (C2 T1) → **recommend immediately**. Role + seniority + context is enough.
- A full JD pasted in (C9 T1) → **clarify if the JD has multiple competing focuses** (backend vs frontend vs balanced). One JD, multiple plausible batteries → ask which one.
- "I need an assessment" (spec example) → **clarify**.

So the heuristic is roughly: clarify if the dimension of variation in the catalog (test type, duration, level) cannot be pinned down to a coherent slice from what the user gave.

### Refinement

- "Add X" → keep previous, add new items matching X (C4, C9).
- "Drop X / Remove X" → filter previous (C9 T4, C10 T2).
- "Replace X with something shorter/cheaper" → swap (C10 T2).
- The final shortlist must remain a coherent battery; the agent doesn't blindly do what the user asks if it produces an empty list. (No trace shows this edge case explicitly, but it's implied.)

### Compare

- Compare questions arrive **mid-conversation** after a shortlist already exists (C3 T4, C5 T2, C6 T2).
- The agent answers grounded in catalog descriptions and **does not change the shortlist** unless the user follows up with a refinement.
- The reply for a compare is prose, not a table; the previous shortlist stays current.

### Refuse / soft-decline

- C7 T3: user asks a legal/HIPAA compliance question. Agent says "that's outside what I can advise on — counsel can answer that. What I CAN confirm: [factual statement from the catalog]." Then keeps the previous shortlist intact.
- This is a **soft decline**: refuses the off-topic sub-question, stays engaged on assessments. NOT a full refusal that ends the conversation.
- Pure off-topic / prompt-injection → would be a full refuse (no traces of this; behavior probes will test it).

### end_of_conversation

- Set to `true` only after the user signals satisfaction: "Perfect", "Confirmed", "Locking it in", "That covers it."
- Until then, even after the agent has committed to a final-looking shortlist, EOC stays `false`.
- The agent never sets EOC `true` unilaterally.

### Turn-budget discipline

- C9 used 7 of 8 turns. Tight.
- Two clarify turns is the realistic ceiling before committing.
- On turn 7 with no commit yet, force a recommend with whatever context is available.

## Soft-decline pattern (NEW node not in initial scaffold)

I had a single `refuse` node. The traces show that's wrong — there's a soft decline (decline this sub-question, keep the shortlist) AND a hard refuse (off-topic / injection, no shortlist exists). Two paths:

- **Soft decline + re-emit**: legal/compliance/policy questions when a shortlist already exists. Decline the sub-question, re-emit the existing shortlist or leave it implied.
- **Hard refuse**: prompt injection, completely off-topic, no shortlist to fall back on. Decline politely, no recommendations.

## Test_type field bug I almost shipped

I had `test_type_str` joining with spaces (`"A B P"`). Trace format is commas (`"A,B,P"`). Fixed in app/retrieval/catalog.py.
