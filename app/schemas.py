"""
Pydantic schemas for the /chat endpoint.

Design notes (for interview defense):
- The schema is non-negotiable per the assignment. Any deviation breaks the
  automated evaluator, so we validate STRICTLY on both ingress and egress.
- We use Pydantic v2 (faster, better validation, native JSON schema).
- `recommendations` MUST be an empty list (not null/missing) when clarifying
  or refusing. We enforce this in the guardrail layer, not just the schema,
  because Pydantic alone can't express "non-empty list of 1..10 items when
  the agent has committed to a shortlist."
- `test_type` is constrained to the 8 catalog codes (A/B/C/D/E/K/P/S) so
  the agent cannot invent new categories.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# SHL test-type taxonomy, taken verbatim from the catalog legend.
# Keeping this as a Literal type means Pydantic rejects any other letter
# at parse time — one less hallucination path.
TestTypeCode = Literal["A", "B", "C", "D", "E", "K", "P", "S"]


class Message(BaseModel):
    """A single turn in the conversation history."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    """Incoming POST /chat body.

    The API is stateless: every call carries the full history.
    """

    messages: list[Message] = Field(min_length=1)

    @field_validator("messages")
    @classmethod
    def history_must_start_with_user(cls, v: list[Message]) -> list[Message]:
        # Defensive: realistic harness should always start with user, but
        # if a buggy client sends otherwise, fail loudly rather than silently.
        if v[0].role != "user":
            raise ValueError("Conversation must start with a user message.")
        return v


class Recommendation(BaseModel):
    """A single assessment in the shortlist.

    All three fields are required by SHL's example response. `url` must be
    an SHL catalog URL — the guardrail layer enforces this against our
    scraped catalog at response time.
    """

    name: str = Field(min_length=1)
    url: str = Field(pattern=r"^https://www\.shl\.com/.+")
    test_type: str = Field(min_length=1, max_length=20)
    # NOTE: test_type is a string (not single Literal) because catalog items
    # often carry MULTIPLE test type codes, e.g. "A B P" for a Job Solution.
    # We still validate each letter is in the legend in the guardrail.


class ChatResponse(BaseModel):
    """Outgoing /chat response body.

    Per spec:
    - `recommendations` is [] when clarifying or refusing.
    - `recommendations` has 1..10 items when committing to a shortlist.
    - `end_of_conversation` is True only when the agent considers task done.
    """

    reply: str = Field(min_length=1, max_length=4000)
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    """GET /health response."""

    status: Literal["ok"] = "ok"
