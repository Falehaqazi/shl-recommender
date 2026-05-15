"""
Agent nodes. Each function is a small, testable unit doing ONE thing.

The orchestration logic lives in graph.py; this file is just the
business logic of each step.

Shared conventions:
- Every node receives the AgentState dict, returns a partial state
  update (LangGraph merges it).
- LLM calls go through llm_client.chat_json. The response is already
  parsed JSON; if it's malformed JSON, llm_client raises and the
  build_response node catches and falls back to a safe reply.
- No node directly constructs the final ChatResponse — that's
  build_response's job. Nodes just populate the in-flight state.

Why split this from graph.py?
- Easier unit tests: you can call extract_previous_shortlist() in a
  pytest with a hand-crafted state dict, no LangGraph runtime needed.
- Easier prompt iteration: change a node without touching the graph.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.prompts import (
    CLARIFY_SYSTEM,
    COMPARE_SYSTEM,
    RECOMMEND_SYSTEM,
    REFINE_SYSTEM,
    REFUSE_SYSTEM,
    ROUTER_SYSTEM,
    SOFT_DECLINE_SYSTEM,
)
from app.config import settings
from app.llm import LLMError, llm_client
from app.retrieval.catalog import AssessmentItem, Catalog
from app.retrieval.index import HybridRetriever, RetrievalHit
from app.schemas import Message, Recommendation

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _history_as_text(messages: list[Message]) -> str:
    """Render conversation history as a readable transcript for prompts."""
    lines = []
    for m in messages:
        prefix = "USER" if m.role == "user" else "ASSISTANT"
        lines.append(f"{prefix}: {m.content}")
    return "\n".join(lines)


def _history_as_chat(messages: list[Message]) -> list[dict[str, str]]:
    """Render history as the messages list the LLM API expects."""
    return [{"role": m.role, "content": m.content} for m in messages]


def _last_user_message(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


# ---------------------------------------------------------------------------
# Extracting previous shortlist from history
# ---------------------------------------------------------------------------

# We mark our own assistant messages with a sentinel comment so we can
# parse a previous shortlist out of the conversation. The sentinel is
# invisible to a human reader because we put it at the end of the message
# after a couple of newlines.
#
# Format: <!--SHL_REC: ["url1","url2",...]-->
#
# This is a stateless API but the evaluator carries history forward, so
# the sentinel travels with the conversation and we can recover the prior
# shortlist on every turn.
SHORTLIST_SENTINEL_RE = re.compile(r"<!--SHL_REC:\s*(\[.*?\])\s*-->", re.DOTALL)


def extract_previous_shortlist(messages: list[Message], catalog: Catalog) -> list[AssessmentItem]:
    """Scan history from newest to oldest for the most recent shortlist.

    Returns [] if no prior shortlist found.
    """
    for m in reversed(messages):
        if m.role != "assistant":
            continue
        match = SHORTLIST_SENTINEL_RE.search(m.content)
        if not match:
            continue
        try:
            urls = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        items: list[AssessmentItem] = []
        for u in urls:
            it = catalog.by_url(u)
            if it is not None:
                items.append(it)
        return items
    return []


def attach_shortlist_sentinel(reply: str, urls: list[str]) -> str:
    """Append the sentinel so future turns can recover this shortlist."""
    if not urls:
        return reply
    payload = json.dumps(urls)
    return f"{reply}\n\n<!--SHL_REC: {payload}-->"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

# Patterns to detect "obviously closure" without an LLM call.
# These are the satisfaction phrases that map 1:1 to user_satisfied=true.
# Keep tight: false positives here = premature end_of_conversation.
_CLOSURE_PATTERNS = [
    r"\bperfect\b.{0,20}\b(that.?s? what|exactly|thanks)\b",
    r"\bconfirmed\b",
    r"\blocking it in\b",
    r"\bthat covers it\b",
    r"\bthat works\b",
    r"\bfinal list\b",
    r"\blgtm\b",
    r"\bsounds good\b.{0,20}(thanks|confirmed)?",
    r"\bwe.?ll go with (that|those)\b",
]
_CLOSURE_RE = re.compile("|".join(_CLOSURE_PATTERNS), re.IGNORECASE)

# Edit-words that indicate refinement intent and MUST NOT short-circuit
# to closure even if a positive word also appears.
_EDIT_RE = re.compile(
    r"\b(add|drop|remove|replace|swap|change|modify|shorter|longer|cheaper|"
    r"more|less|instead|also|but)\b",
    re.IGNORECASE,
)

# Off-topic / injection keywords. Cheap to catch; saves an LLM call.
_INJECTION_RE = re.compile(
    r"\b(ignore (previous|prior|above)|disregard.{0,30}instructions|"
    r"you are now|new instructions|system prompt|jailbreak)\b",
    re.IGNORECASE,
)

# Compare-question detector: "vs", "versus", "difference between", "compared to".
_COMPARE_RE = re.compile(
    r"\b(vs\.?|versus|difference between|what.?s? the diff|compared? to|"
    r"how does .{1,40} compare)\b",
    re.IGNORECASE,
)


def _heuristic_route(
    last_user: str,
    has_prior_shortlist: bool,
    turn_count: int,
) -> dict[str, Any] | None:
    """Try to classify the intent without an LLM call.

    Returns a partial route dict on success, None to fall through to the
    LLM router. We're conservative: only short-circuit when the signal is
    unambiguous.
    """
    lu = last_user.strip()
    if not lu:
        return None

    has_edit_word = bool(_EDIT_RE.search(lu))
    has_closure_phrase = bool(_CLOSURE_RE.search(lu))

    # 1. Pure closure: short message AND closure phrase AND no edit verbs
    #    AND a prior shortlist exists. Skips both router AND node calls
    #    because build_response just re-emits the prior shortlist.
    if (
        has_prior_shortlist
        and has_closure_phrase
        and not has_edit_word
        and len(lu) <= 120
    ):
        return {
            "intent": "refine",  # Will keep prior shortlist unchanged
            "user_satisfied": True,
            "router_reasoning": "heuristic: closure phrase, no edit verbs",
        }

    # 2. Injection attempts: short-circuit straight to refuse.
    if _INJECTION_RE.search(lu):
        return {
            "intent": "refuse",
            "user_satisfied": False,
            "router_reasoning": "heuristic: injection pattern",
        }

    # 3. Compare questions with a prior shortlist: route to compare.
    #    Without prior shortlist we still let the LLM decide (could be a
    #    fresh "compare X vs Y" recommend-ish request).
    if has_prior_shortlist and _COMPARE_RE.search(lu):
        return {
            "intent": "compare",
            "user_satisfied": False,
            "router_reasoning": "heuristic: compare phrasing with prior shortlist",
        }

    return None


def route_intent(state: dict[str, Any]) -> dict[str, Any]:
    """Classify the next agent action.

    Tries cheap heuristics first; falls through to the LLM router only
    when intent is ambiguous. Saves ~25-30% of LLM calls during eval.
    """
    messages: list[Message] = state["messages"]
    transcript = _history_as_text(messages)
    turn_count = sum(1 for m in messages if m.role == "user")
    prior_items = extract_previous_shortlist(messages, state["catalog"])
    has_prior_shortlist = len(prior_items) > 0

    # --- Heuristic short-circuit ---
    last_user = _last_user_message(messages)
    heuristic = _heuristic_route(last_user, has_prior_shortlist, turn_count)
    if heuristic is not None:
        log.info("Heuristic route: %s (%s)", heuristic["intent"], heuristic["router_reasoning"])
        return {
            **heuristic,
            "turn_count": turn_count,
            "prior_shortlist": prior_items,
        }

    # --- LLM router for the ambiguous cases ---
    user_msg = (
        f"CONVERSATION HISTORY:\n{transcript}\n\n"
        f"TURN COUNT (user turns so far): {turn_count}\n"
        f"PRIOR SHORTLIST EXISTS: {has_prior_shortlist}\n\n"
        "Classify the next agent action. Return only the JSON."
    )

    try:
        parsed = llm_client.chat_json(
            system=ROUTER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=200,
        )
        intent = parsed.get("intent", "clarify")
        user_satisfied = bool(parsed.get("user_satisfied", False))
        reasoning = parsed.get("reasoning", "")
    except (LLMError, json.JSONDecodeError) as e:
        log.warning("Router failed (%s); defaulting to clarify", e)
        intent = "clarify"
        user_satisfied = False
        reasoning = f"router fallback: {e}"

    # Sanity-correct: refine without prior shortlist -> recommend.
    if intent == "refine" and not has_prior_shortlist:
        log.info("Router said refine but no prior shortlist; switching to recommend")
        intent = "recommend"

    # Turn-budget enforcement: on turn 4+ with no commit, force recommend
    # unless the user is asking to compare / decline.
    if turn_count >= settings.max_clarifications + 2 and intent == "clarify":
        log.info("Turn budget tight (turn=%d); forcing recommend", turn_count)
        intent = "recommend"

    return {
        "intent": intent,
        "user_satisfied": user_satisfied,
        "router_reasoning": reasoning,
        "turn_count": turn_count,
        "prior_shortlist": prior_items,
    }


# ---------------------------------------------------------------------------
# CLARIFY
# ---------------------------------------------------------------------------

def node_clarify(state: dict[str, Any]) -> dict[str, Any]:
    transcript = _history_as_text(state["messages"])
    user_msg = f"CONVERSATION HISTORY:\n{transcript}\n\nAsk one clarifying question."
    try:
        parsed = llm_client.chat_json(
            system=CLARIFY_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.2,
            max_tokens=120,
        )
        reply = (parsed.get("reply") or "").strip()
    except (LLMError, json.JSONDecodeError) as e:
        log.warning("Clarify failed: %s", e)
        reply = "Could you tell me a bit more about the role and seniority level you're hiring for?"

    if not reply:
        reply = "Could you tell me a bit more about the role you're hiring for?"

    return {"reply": reply, "selected_items": []}


# ---------------------------------------------------------------------------
# RECOMMEND
# ---------------------------------------------------------------------------

def _format_candidates(hits: list[RetrievalHit]) -> str:
    """Render retrieved candidates as a compact list the LLM can pick from."""
    lines = []
    for i, h in enumerate(hits, 1):
        it = h.item
        duration = f"{it.duration_minutes}min" if it.duration_minutes else "—"
        lines.append(
            f"{i}. {it.name}\n"
            f"   url: {it.url}\n"
            f"   test_type: {','.join(it.test_type)}\n"
            f"   duration: {duration}\n"
            f"   adaptive: {'yes' if it.adaptive else 'no'}\n"
            f"   description: {it.description[:400]}"
        )
    return "\n\n".join(lines)


def node_recommend(state: dict[str, Any]) -> dict[str, Any]:
    """Retrieve candidates, then have the LLM pick a shortlist from them."""
    retriever: HybridRetriever = state["retriever"]
    catalog: Catalog = state["catalog"]
    messages: list[Message] = state["messages"]

    # Build retrieval query: concatenate all user turns so context accumulates.
    user_turns = [m.content for m in messages if m.role == "user"]
    query = " ".join(user_turns)

    # Wider candidate pool than final_top_k so the LLM has room to filter.
    hits = retriever.search(query, top_k=settings.retrieval_top_k)
    if not hits:
        return {"reply": "I couldn't find good matches for that request. Could you tell me more about the role or skills you're hiring for?", "selected_items": []}

    candidates_str = _format_candidates(hits)
    transcript = _history_as_text(messages)
    user_msg = (
        f"CONVERSATION:\n{transcript}\n\n"
        f"CANDIDATE LIST (from retrieval):\n{candidates_str}\n\n"
        "Pick 1-10 of these. Return JSON with reply + selected_urls."
    )

    try:
        parsed = llm_client.chat_json(
            system=RECOMMEND_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.1,
            max_tokens=700,
        )
        reply = (parsed.get("reply") or "").strip()
        urls = parsed.get("selected_urls") or []
    except (LLMError, json.JSONDecodeError) as e:
        log.warning("Recommend failed: %s", e)
        # Fallback: return top retrieval hits without LLM rerank.
        urls = [h.item.url for h in hits[: settings.final_top_k]]
        reply = "Here are the most relevant assessments for your request."

    # Resolve URLs -> items. Drop any URL not in the catalog (guardrail).
    selected_items: list[AssessmentItem] = []
    seen_urls: set[str] = set()
    for u in urls:
        if u in seen_urls:
            continue
        it = catalog.by_url(u)
        if it is not None:
            selected_items.append(it)
            seen_urls.add(u)
        if len(selected_items) >= 10:
            break

    # If the LLM whiffed entirely, fall back to top hits.
    if not selected_items:
        log.warning("LLM picked no valid URLs; falling back to retrieval top-k")
        for h in hits[: settings.final_top_k]:
            if h.item.url not in seen_urls:
                selected_items.append(h.item)
                seen_urls.add(h.item.url)

    if not reply:
        reply = f"Here are {len(selected_items)} assessments that fit."

    return {"reply": reply, "selected_items": selected_items}


# ---------------------------------------------------------------------------
# REFINE
# ---------------------------------------------------------------------------

def node_refine(state: dict[str, Any]) -> dict[str, Any]:
    retriever: HybridRetriever = state["retriever"]
    catalog: Catalog = state["catalog"]
    messages: list[Message] = state["messages"]
    prior: list[AssessmentItem] = state.get("prior_shortlist") or []

    # Fast path: heuristic router classified this as pure closure
    # (user_satisfied=True + prior shortlist exists). Skip LLM call,
    # re-emit prior unchanged. build_response sets end_of_conversation.
    if state.get("user_satisfied") and prior:
        log.info("Refine fast-path: closure detected, re-emitting prior shortlist")
        return {
            "reply": "Confirmed. Final shortlist as discussed.",
            "selected_items": list(prior),
        }

    # Retrieve fresh candidates for whatever the user's edit asks for.
    last_user = _last_user_message(messages)
    hits = retriever.search(last_user, top_k=settings.retrieval_top_k)

    previous_str = "\n".join(
        f"- {it.name} | {it.url} | {','.join(it.test_type)} | "
        f"{'%dmin' % it.duration_minutes if it.duration_minutes else '—'}"
        for it in prior
    ) or "(no previous shortlist)"

    candidates_str = _format_candidates(hits)

    transcript = _history_as_text(messages)
    user_msg = (
        f"CONVERSATION:\n{transcript}\n\n"
        f"PREVIOUS:\n{previous_str}\n\n"
        f"CANDIDATES:\n{candidates_str}\n\n"
        "Apply the user's edit. Return JSON with reply + selected_urls."
    )

    try:
        parsed = llm_client.chat_json(
            system=REFINE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.1,
            max_tokens=700,
        )
        reply = (parsed.get("reply") or "").strip()
        urls = parsed.get("selected_urls") or []
    except (LLMError, json.JSONDecodeError) as e:
        log.warning("Refine failed: %s", e)
        # Fallback: keep previous shortlist unchanged.
        urls = [it.url for it in prior]
        reply = "Keeping the previous shortlist."

    # Resolve URLs to items, preserving order.
    allowed_urls = {it.url for it in prior} | {h.item.url for h in hits}
    selected_items: list[AssessmentItem] = []
    seen: set[str] = set()
    for u in urls:
        if u in seen or u not in allowed_urls:
            continue
        it = catalog.by_url(u)
        if it is not None:
            selected_items.append(it)
            seen.add(u)
        if len(selected_items) >= 10:
            break

    # If LLM produced nothing valid, keep prior unchanged.
    if not selected_items:
        log.warning("Refine produced no valid URLs; keeping prior")
        selected_items = list(prior)
        reply = reply or "Keeping the previous shortlist."

    if not reply:
        reply = "Updated shortlist."

    return {"reply": reply, "selected_items": selected_items}


# ---------------------------------------------------------------------------
# COMPARE
# ---------------------------------------------------------------------------

# Words that often appear next to assessment names so we can strip them.
_COMPARE_NOISE = re.compile(r"\b(test|tests|assessment|assessments|the|a|an|and|or|vs|versus|between)\b", re.I)


def _extract_assessment_names(text: str, catalog: Catalog) -> list[AssessmentItem]:
    """Heuristic: pull catalog names mentioned in the user message.

    We rank candidate items by length of name match. Longer matches win
    so "OPQ MQ Sales Report" beats just "OPQ" when both could match.
    """
    text_lower = text.lower()
    matches: list[tuple[int, AssessmentItem]] = []
    for it in catalog.items:
        # Match against full name AND key tokens like "OPQ32r", "DSI", "GSA".
        candidates = [it.name.lower()]
        # Pull capitalized acronyms / version tokens from the name.
        for tok in re.findall(r"\b[A-Z][A-Z0-9+]+\b", it.name):
            if len(tok) >= 2:
                candidates.append(tok.lower())
        for c in candidates:
            if c and c in text_lower:
                matches.append((len(c), it))
                break

    # Sort by match length descending, dedupe by URL.
    matches.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    out: list[AssessmentItem] = []
    for _, it in matches:
        if it.url not in seen:
            out.append(it)
            seen.add(it.url)
    return out[:5]  # cap


def node_compare(state: dict[str, Any]) -> dict[str, Any]:
    catalog: Catalog = state["catalog"]
    messages: list[Message] = state["messages"]
    prior: list[AssessmentItem] = state.get("prior_shortlist") or []

    last_user = _last_user_message(messages)
    # Look in last user message AND prior shortlist for named items.
    named = _extract_assessment_names(last_user, catalog)
    if not named:
        # Fall back: use prior shortlist as the compare set.
        named = prior[:5]

    if not named:
        return {
            "reply": "I'm not sure which assessments you'd like to compare. Could you name them?",
            "selected_items": list(prior),  # keep prior shortlist unchanged
        }

    entries_str = "\n\n".join(
        f"NAME: {it.name}\n"
        f"URL: {it.url}\n"
        f"TEST_TYPE: {','.join(it.test_type)}\n"
        f"JOB_LEVELS: {', '.join(it.job_levels) or '—'}\n"
        f"DURATION: {it.duration_minutes if it.duration_minutes else '—'} min\n"
        f"LANGUAGES: {', '.join(it.languages[:6]) or '—'}\n"
        f"ADAPTIVE: {'yes' if it.adaptive else 'no'}\n"
        f"DESCRIPTION: {it.description}"
        for it in named
    )

    transcript = _history_as_text(messages)
    user_msg = (
        f"CONVERSATION:\n{transcript}\n\n"
        f"CATALOG_ENTRIES:\n{entries_str}\n\n"
        "Compare these. Return JSON with just reply."
    )

    try:
        parsed = llm_client.chat_json(
            system=COMPARE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.2,
            max_tokens=500,
        )
        reply = (parsed.get("reply") or "").strip()
    except (LLMError, json.JSONDecodeError) as e:
        log.warning("Compare failed: %s", e)
        reply = "I can compare those, but I'm having trouble pulling the details right now. Could you rephrase?"

    # Compare preserves the prior shortlist; the evaluator sees the same items.
    return {"reply": reply, "selected_items": list(prior)}


# ---------------------------------------------------------------------------
# SOFT DECLINE
# ---------------------------------------------------------------------------

def node_soft_decline(state: dict[str, Any]) -> dict[str, Any]:
    messages: list[Message] = state["messages"]
    prior: list[AssessmentItem] = state.get("prior_shortlist") or []
    catalog: Catalog = state["catalog"]

    # Provide the LLM with a small slice of catalog context (items the user
    # might be referring to) so the factual-pivot can be grounded.
    last_user = _last_user_message(messages)
    relevant_items = _extract_assessment_names(last_user, catalog)
    # Also include prior shortlist items.
    for it in prior:
        if it not in relevant_items:
            relevant_items.append(it)
    relevant_items = relevant_items[:5]

    context_str = "\n".join(
        f"- {it.name}: {it.description[:300]}"
        for it in relevant_items
    ) or "(no specific catalog context)"

    transcript = _history_as_text(messages)
    user_msg = (
        f"CONVERSATION:\n{transcript}\n\n"
        f"RELEVANT CATALOG ITEMS (for factual pivot only):\n{context_str}\n\n"
        "Decline the off-scope sub-question and pivot to catalog facts. Return JSON."
    )

    try:
        parsed = llm_client.chat_json(
            system=SOFT_DECLINE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.2,
            max_tokens=300,
        )
        reply = (parsed.get("reply") or "").strip()
    except (LLMError, json.JSONDecodeError) as e:
        log.warning("Soft decline failed: %s", e)
        reply = "That's outside what I can advise on — your legal or compliance team is the right resource. I can keep helping with SHL assessment selection."

    if not reply:
        reply = "That falls outside what I can advise on. Happy to keep helping with SHL assessment selection."

    return {"reply": reply, "selected_items": list(prior)}


# ---------------------------------------------------------------------------
# REFUSE (hard)
# ---------------------------------------------------------------------------

def node_refuse(state: dict[str, Any]) -> dict[str, Any]:
    transcript = _history_as_text(state["messages"])
    user_msg = f"CONVERSATION:\n{transcript}\n\nDecline politely. Return JSON."
    try:
        parsed = llm_client.chat_json(
            system=REFUSE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=120,
        )
        reply = (parsed.get("reply") or "").strip()
    except (LLMError, json.JSONDecodeError) as e:
        log.warning("Refuse failed: %s", e)
        reply = "I can only help with SHL assessment selection. What role are you hiring for?"

    if not reply:
        reply = "I can only help with SHL assessment selection. What role are you hiring for?"

    # Hard refuse: drop any prior shortlist entirely. We are saying "no".
    return {"reply": reply, "selected_items": []}
