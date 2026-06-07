"""Tests for the cascade proxy's single-token logprob classification path.

These exercise ``APIWrapper.classify_with_logprob`` and its pure parser
``_parse_logprob_response`` without any real LLM calls: responses are synthetic
SimpleNamespace objects shaped like litellm's, and the module-level
``completion`` is monkeypatched.
"""

import math
from types import SimpleNamespace

import pytest

import docetl.operations.utils.api as api_mod
from docetl.operations.utils.api import APIWrapper


# ---------------------------------------------------------------------------
# Helpers: build litellm-shaped logprob responses.
# ---------------------------------------------------------------------------
def _alt(token, logprob):
    return SimpleNamespace(token=token, logprob=logprob)


def make_response(chosen_token, top_logprobs):
    """A response whose first generated token carries ``top_logprobs`` alts."""
    first = SimpleNamespace(
        token=chosen_token,
        logprob=top_logprobs[0].logprob if top_logprobs else 0.0,
        top_logprobs=top_logprobs,
    )
    logprobs = SimpleNamespace(content=[first])
    choice = SimpleNamespace(logprobs=logprobs)
    return SimpleNamespace(choices=[choice])


class FakeRunner:
    """Minimal runner satisfying APIWrapper's constructor + proxy path."""

    def __init__(self):
        self.config = {}
        self.is_cancelled = False

    def blocking_acquire(self, *args, **kwargs):
        pass


@pytest.fixture
def wrapper():
    return APIWrapper(FakeRunner())


# ---------------------------------------------------------------------------
# Pure parser: _parse_logprob_response
# ---------------------------------------------------------------------------
def test_parse_picks_argmax_label():
    # "1" much more likely than "2" -> label True with high confidence.
    resp = make_response("1", [_alt("1", math.log(0.9)), _alt("2", math.log(0.1))])
    label, prob = APIWrapper._parse_logprob_response(resp, {"1": True, "2": False})
    assert label is True
    assert prob == pytest.approx(0.9, abs=1e-9)


def test_parse_uses_raw_logprob_ignoring_off_menu_tokens():
    # A non-menu token ("the") in the alternatives is ignored; the chosen
    # label's raw model probability is returned (not renormalized).
    resp = make_response(
        "2",
        [
            _alt("the", math.log(0.5)),
            _alt("1", math.log(0.2)),
            _alt("2", math.log(0.3)),
        ],
    )
    label, prob = APIWrapper._parse_logprob_response(resp, {"1": "a", "2": "b"})
    assert label == "b"
    assert prob == pytest.approx(0.3, abs=1e-9)


def test_parse_strips_whitespace_on_tokens():
    resp = make_response(" 1", [_alt(" 1", math.log(0.8)), _alt("2", math.log(0.2))])
    label, prob = APIWrapper._parse_logprob_response(resp, {"1": "yes", "2": "no"})
    assert label == "yes"
    assert prob == pytest.approx(0.8, abs=1e-9)


def test_parse_falls_back_to_chosen_token_without_top_logprobs():
    # Provider returned no alternatives -> use the chosen token alone.
    first = SimpleNamespace(token="2", logprob=math.log(0.7), top_logprobs=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[first]))]
    )
    label, prob = APIWrapper._parse_logprob_response(resp, {"1": "a", "2": "b"})
    assert label == "b"
    assert prob == pytest.approx(0.7, abs=1e-9)


def test_parse_raises_when_no_logprobs():
    resp = SimpleNamespace(choices=[SimpleNamespace(logprobs=None)])
    with pytest.raises(ValueError, match="logprobs"):
        APIWrapper._parse_logprob_response(resp, {"1": True, "2": False})


def test_parse_raises_when_answer_off_menu():
    resp = make_response("foo", [_alt("foo", math.log(0.9))])
    with pytest.raises(ValueError, match="did not match any menu option"):
        APIWrapper._parse_logprob_response(resp, {"1": True, "2": False})


# ---------------------------------------------------------------------------
# classify_with_logprob: input validation
# ---------------------------------------------------------------------------
def test_empty_labels_raises(wrapper):
    with pytest.raises(ValueError, match="non-empty"):
        wrapper.classify_with_logprob("gpt-4o-mini", [{"role": "user", "content": "x"}], [])


def test_too_many_labels_raises(wrapper):
    with pytest.raises(ValueError, match="at most 9"):
        wrapper.classify_with_logprob(
            "gpt-4o-mini", [{"role": "user", "content": "x"}], list(range(10))
        )


# ---------------------------------------------------------------------------
# classify_with_logprob: end-to-end with a monkeypatched completion
# ---------------------------------------------------------------------------
def test_classify_builds_menu_and_returns_label(wrapper, monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return make_response("2", [_alt("1", math.log(0.25)), _alt("2", math.log(0.75))])

    monkeypatch.setattr(api_mod, "completion", fake_completion)

    label, prob = wrapper.classify_with_logprob(
        "gpt-4o-mini",
        [{"role": "user", "content": "Is the sky green?"}],
        [True, False],
    )

    assert label is False
    assert prob == pytest.approx(0.75, abs=1e-9)

    # logprobs were requested and decoding constrained to a single token.
    assert captured["logprobs"] is True
    assert captured["max_tokens"] == 1
    assert captured["temperature"] == 0

    # The menu instruction was appended as the final user message.
    last = captured["messages"][-1]
    assert last["role"] == "user"
    assert "1 = True" in last["content"]
    assert "2 = False" in last["content"]
    # The caller's message is preserved between the system prompt and the menu.
    assert any(m["content"] == "Is the sky green?" for m in captured["messages"])


def test_classify_top_logprobs_sized_to_labels(wrapper, monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return make_response("1", [_alt("1", math.log(0.99)), _alt("2", math.log(0.01))])

    monkeypatch.setattr(api_mod, "completion", fake_completion)
    wrapper.classify_with_logprob(
        "gpt-4o-mini", [{"role": "user", "content": "x"}], ["a", "b"]
    )
    # Default floor is 10, capped at 20.
    assert captured["top_logprobs"] == 10


def test_classify_respects_explicit_top_logprobs_and_system_prompt(wrapper, monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return make_response("1", [_alt("1", math.log(0.6)), _alt("2", math.log(0.4))])

    monkeypatch.setattr(api_mod, "completion", fake_completion)
    wrapper.classify_with_logprob(
        "gpt-4o-mini",
        [{"role": "user", "content": "x"}],
        ["a", "b"],
        top_logprobs=5,
        system_prompt="custom classifier",
    )
    assert captured["top_logprobs"] == 5
    assert captured["messages"][0] == {"role": "system", "content": "custom classifier"}
