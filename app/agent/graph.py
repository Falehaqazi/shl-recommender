"""
LangGraph state machine for the SHL recommender agent.

Flow:

    START
      |
      v
    route_intent  (LLM router classifies into one of 6 intents)
      |
      +--> clarify ------+
      +--> recommend ----+
      +--> refine -------+
      +--> compare ------+
      +--> soft_decline -+
      +--> refuse -------+
                         |
                         v
                  build_response  (guardrail layer)
                         |
                         v
                        END

Why a graph rather than if/elif chains?
- Each node is independently testable.
- Adding a new intent is a node + a router edge; no central function grows.
- LangGraph traces give us a free debug log of which path each request took
  — invaluable during eval iteration.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    attach_shortlist_sentinel,
    node_clarify,
    node_compare,
    node_recommend,
    node_refine,
    node_refuse,
    node_soft_decline,
    route_intent,
)
from app.retrieval.catalog import AssessmentItem, Catalog
from app.retrieval.index import HybridRetriever
from app.schemas import ChatResponse, Message, Recommendation

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    # Inputs (immutable per turn)
    messages: list[Message]
    catalog: Catalog
    retriever: HybridRetriever

    # Router output
    intent: str
    user_satisfied: bool
    router_reasoning: str
    turn_count: int
    prior_shortlist: list[AssessmentItem]

    # Node outputs
    reply: str
    selected_items: list[AssessmentItem]

    # Final response (built by build_response)
    response: ChatResponse


# ---------------------------------------------------------------------------
# Final guardrail node
# ---------------------------------------------------------------------------

def build_response(state: AgentState) -> dict[str, Any]:
    """Assemble ChatResponse with all final guardrails.

    Guarantees:
    - recommendations only contains URLs that exist in the catalog (re-checked)
    - recommendations has 1-10 items OR is empty (never >10)
    - test_type values are subsets of {A,B,C,D,E,K,P,S}
    - end_of_conversation is True only when user_satisfied AND a shortlist exists
    - reply is non-empty
    - the reply carries a sentinel comment so future turns can recover this shortlist
    """
    catalog: Catalog = state["catalog"]
    reply: str = state.get("reply") or "Got it."
    selected_items: list[AssessmentItem] = state.get("selected_items") or []
    intent: str = state.get("intent", "clarify")
    user_satisfied: bool = state.get("user_satisfied", False)

    # --- Validate items against catalog. Drop any that don't resolve.
    valid_items: list[AssessmentItem] = []
    seen_urls: set[str] = set()
    for it in selected_items:
        if it.url in seen_urls:
            continue
        if not catalog.has_url(it.url):
            log.warning("Guardrail dropped non-catalog URL: %s", it.url)
            continue
        valid_items.append(it)
        seen_urls.add(it.url)

    # --- Enforce 1-10 cap. (Schema also enforces, but we trim defensively.)
    if len(valid_items) > 10:
        valid_items = valid_items[:10]

    # --- Build Recommendation objects.
    recommendations: list[Recommendation] = []
    for it in valid_items:
        # test_type letters: filter to known codes only.
        letters = [t for t in it.test_type if t in {"A", "B", "C", "D", "E", "K", "P", "S"}]
        if not letters:
            log.warning("Item has no valid test_type letters: %s", it.url)
            # Skip rather than emit an invalid Recommendation.
            continue
        recommendations.append(
            Recommendation(
                name=it.name,
                url=it.url,
                test_type=",".join(letters),
            )
        )

    # --- end_of_conversation logic.
    # True only when: user signaled satisfaction AND we have committed to a
    # shortlist (this turn, or by carrying the prior shortlist on a compare/
    # soft_decline turn).
    end_of_conversation = bool(user_satisfied and recommendations)

    # --- Attach sentinel to reply so the next turn can recover this shortlist.
    reply_with_sentinel = attach_shortlist_sentinel(
        reply.strip(), [r.url for r in recommendations]
    )

    # --- For "clarify" / "refuse" intents, there must be no recommendations.
    # (Hard refuse is enforced in node_refuse. Clarify is enforced here.)
    if intent in {"clarify", "refuse"}:
        recommendations = []
        # Don't emit sentinel for empty shortlist.
        reply_with_sentinel = reply.strip()

    response = ChatResponse(
        reply=reply_with_sentinel,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
    return {"response": response}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _route_edge(state: AgentState) -> str:
    """Return the name of the next node based on intent."""
    intent = state.get("intent", "clarify")
    return {
        "clarify": "clarify",
        "recommend": "recommend",
        "refine": "refine",
        "compare": "compare",
        "soft_decline": "soft_decline",
        "refuse": "refuse",
    }.get(intent, "clarify")


def build_graph():
    """Compile the LangGraph. Call once at app startup."""
    g = StateGraph(AgentState)

    g.add_node("route", route_intent)
    g.add_node("clarify", node_clarify)
    g.add_node("recommend", node_recommend)
    g.add_node("refine", node_refine)
    g.add_node("compare", node_compare)
    g.add_node("soft_decline", node_soft_decline)
    g.add_node("refuse", node_refuse)
    g.add_node("build_response", build_response)

    g.set_entry_point("route")
    g.add_conditional_edges(
        "route",
        _route_edge,
        {
            "clarify": "clarify",
            "recommend": "recommend",
            "refine": "refine",
            "compare": "compare",
            "soft_decline": "soft_decline",
            "refuse": "refuse",
        },
    )
    for node in ("clarify", "recommend", "refine", "compare", "soft_decline", "refuse"):
        g.add_edge(node, "build_response")
    g.add_edge("build_response", END)

    return g.compile()


# Module-level singleton, lazily built on first use to avoid blocking imports.
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
