"""Exercise the shared transport against a local official-SDK server."""
import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from mcp_client import AsyncMCPClient

SERVER = '''
import asyncio
from mcp.server.fastmcp import FastMCP
server = FastMCP("test")
@server.tool()
async def echo(value: str, delay: float = 0) -> dict:
    await asyncio.sleep(delay)
    return {"value": value}
@server.tool()
def fail() -> dict:
    raise ValueError("test failure")
server.run(transport="stdio")
'''


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_errors_timeouts_and_cancellation(self):
        client = AsyncMCPClient(sys.executable, ["-c", SERVER], None)
        await client.start()
        try:
            self.assertEqual({t.name for t in await client.list_tools()}, {"echo", "fail"})
            self.assertEqual(await client.call_tool("echo", {"value": "Café"}), {"value": "Café"})
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                await client.call_tool("fail", {})
            with patch("mcp_client.TOOL_TIMEOUT_S", .01):
                with self.assertRaisesRegex(TimeoutError, "mcp_tool_timeout_after_"):
                    await client.call_tool("echo", {"value": "slow", "delay": .2})
            request = asyncio.create_task(client.call_tool("echo", {"value": "cancelled", "delay": .05}))
            await asyncio.sleep(.01)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            self.assertEqual(await client.call_tool("echo", {"value": "still running"}), {"value": "still running"})
        finally:
            await client.close()
        with self.assertRaisesRegex(RuntimeError, "not running"):
            await client.list_tools()

    async def test_startup_failure_does_not_leave_waiters(self):
        client = AsyncMCPClient("/nonexistent/mcp-test-command", [], None)
        with self.assertRaisesRegex(RuntimeError, "Failed to start"):
            await client.start()
        await client.close()
