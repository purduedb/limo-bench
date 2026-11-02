"""MCP stdio client for the live server.ts (LIMO_SERVER_DIR: another directory holding server.ts); LATENCY_LOG records every call()."""
import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_DIR = Path(os.getenv("LIMO_SERVER_DIR") or Path(__file__).resolve().parents[1])
SERVER_ENTRY = SERVER_DIR / "server.ts"

LATENCY_LOG: list[tuple[str, str, float]] = []  # (test_nodeid, tool, seconds)
_current_test_id = "unknown"


@asynccontextmanager
async def mcp_session(elicitation_callback=None):
    """Spawn server.ts via tsx and yield a connected ClientSession (elicitation only if a callback is given)."""
    params = StdioServerParameters(
        command="npx",
        args=["tsx", str(SERVER_ENTRY)],
        cwd=str(SERVER_DIR),
        env={**os.environ},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, elicitation_callback=elicitation_callback) as session:
            await session.initialize()
            yield session


def call(session: ClientSession, tool: str, args: dict) -> dict:
    """Call a tool synchronously, return its parsed JSON response (or raise)."""
    # must reuse the fixture's loop, not the ambient one - a mismatch breaks anyio cancel scopes
    loop = getattr(session, "_test_loop", None) or asyncio.get_event_loop()
    t0 = time.perf_counter()
    result = loop.run_until_complete(session.call_tool(tool, args))
    LATENCY_LOG.append((_current_test_id, tool, time.perf_counter() - t0))
    if result.isError:
        text = result.content[0].text if result.content else "(no content)"
        raise RuntimeError(f"tool {tool} returned isError=true: {text}")
    text = result.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise AssertionError(f"tool {tool} did not return valid JSON: {text[:500]}")


ENTITY_ID_RE = re.compile(r"^(point|line|polygon):-?\d+$|^tmp:")


def ref(session, place):
    """The reference argument of a tool call: an entity_id or {lat, lon} as given, free text geocoded to its top entity_id first."""
    if isinstance(place, str) and not ENTITY_ID_RE.match(place):
        return call(session, "geocode", {"query": place, "limit": 1})["locations"][0]["entity_id"]
    return place


UNIT_FACTORS = {"km": 1000, "mi": 1609.344}


def assert_converted(item, name, unit):
    """units adds <name>_<unit> beside the untouched <name>_m; m adds nothing."""
    extra = {k for k in item if k.startswith(name + "_") and k != f"{name}_m"}
    assert extra == (set() if unit == "m" else {f"{name}_{unit}"}), (unit, extra)
    if unit != "m":
        assert abs(item[f"{name}_{unit}"] * UNIT_FACTORS[unit] - item[f"{name}_m"]) < 2e-3, (unit, item)


def place(p):
    """A PlaceRef argument: {entity_id} for an id, a {lat, lon} dict as given."""
    return {"entity_id": p} if isinstance(p, str) else p


def rejects(session, pattern, tool, **args):
    """Assert a tool call fails with an error matching pattern."""
    with pytest.raises(RuntimeError, match=pattern):
        call(session, tool, args)


def error_of(session, tool, **args):
    """The error text of a tool call that must fail."""
    with pytest.raises(RuntimeError) as e:
        call(session, tool, args)
    return str(e.value)


def session_with(elicitation_callback=None):
    """Generator behind the session fixtures: one server.ts subprocess per test module."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _mk():
        cm = mcp_session(elicitation_callback)
        s = await cm.__aenter__()
        return cm, s

    cm, s = loop.run_until_complete(_mk())
    s._test_loop = loop  # let call() reuse this exact loop, never the ambient default
    yield s
    try:
        loop.run_until_complete(cm.__aexit__(None, None, None))
    except RuntimeError as e:
        # anyio cancel-scope teardown across run_until_complete calls; every assertion has already run
        if "cancel scope" not in str(e):
            raise
    loop.close()


@pytest.fixture(scope="module")
def session():
    """Shared pytest fixture: one MCP session (one server.ts subprocess) per test module."""
    yield from session_with()
