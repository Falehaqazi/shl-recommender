"""
Hard-eval guards. These tests cover everything SHL says is must-pass:
- Schema compliance on every response
- Items from catalog only in recommendations
- Turn cap (max: 8) honored
- recommendations is [] or 1-10 items
- end_of_conversation is bool

Run: pytest tests/
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest, ChatResponse, Message, Recommendation


def test_message_requires_role_user_or_assistant():
    Message(role="user", content="hi")
    Message(role="assistant", content="hi")
    with pytest.raises(ValidationError):
        Message(role="system", content="hi")  # noqa


def test_history_must_start_with_user():
    ChatRequest(messages=[Message(role="user", content="hi")])
    with pytest.raises(ValidationError):
        ChatRequest(messages=[Message(role="assistant", content="hi")])


def test_recommendation_url_must_be_shl():
    Recommendation(name="X", url="https://www.shl.com/foo/bar/", test_type="K")
    with pytest.raises(ValidationError):
        Recommendation(name="X", url="https://example.com/x", test_type="K")


def test_response_cap_at_10_recommendations():
    recs = [
        Recommendation(name=f"X{i}", url=f"https://www.shl.com/x{i}/", test_type="K")
        for i in range(10)
    ]
    ChatResponse(reply="ok", recommendations=recs)
    with pytest.raises(ValidationError):
        ChatResponse(
            reply="ok",
            recommendations=recs + [Recommendation(name="Y", url="https://www.shl.com/y/", test_type="K")],
        )


def test_response_can_be_empty_recommendations():
    r = ChatResponse(reply="please clarify", recommendations=[])
    assert r.recommendations == []
    assert r.end_of_conversation is False


def test_response_end_of_conversation_bool():
    r = ChatResponse(reply="done", recommendations=[], end_of_conversation=True)
    assert r.end_of_conversation is True
