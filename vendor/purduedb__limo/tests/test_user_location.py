"""Tests for `get_user_location`: the server relays the position it elicits from the client.

Run: python -m pytest tests/test_user_location.py -v -s
"""
import sys
from pathlib import Path

import pytest
from mcp.types import ElicitResult

sys.path.insert(0, str(Path(__file__).parent))
from harness import call, session_with  # noqa: F401

POSITION = {"lat": 40.4237, "lon": -86.9212, "accuracy_m": 12}


async def _accept(context, params):
    return ElicitResult(action="accept", content=POSITION)


async def _decline(context, params):
    return ElicitResult(action="decline")


@pytest.fixture(scope="module")
def accepting_session():
    yield from session_with(_accept)


@pytest.fixture(scope="module")
def declining_session():
    yield from session_with(_decline)


@pytest.fixture(scope="module")
def plain_session():
    yield from session_with()  # client without elicitation support


def test_user_location_returns_client_position(accepting_session):
    assert call(accepting_session, "get_user_location", {}) == POSITION


def test_user_location_declined(declining_session):
    with pytest.raises(RuntimeError, match="declined"):
        call(declining_session, "get_user_location", {})


def test_user_location_unsupported_client(plain_session):
    with pytest.raises(RuntimeError, match="client cannot provide"):
        call(plain_session, "get_user_location", {})
