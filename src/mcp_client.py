from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional


TOOL_TIMEOUT_S = 10
LOCATION_TIMEOUT_S = 40


class AsyncMCPClient:
    """Serialize requests in the task that owns the SDK's task-scoped session."""

    def __init__(
        self,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]],
        cwd: Optional[str] = None,
        elicitation_callback: Any = None,
    ) -> None:
        self._command = command
        self._args = args
        self._env = env
        self._cwd = cwd
        self._elicitation_callback = elicitation_callback
        self._requests: "asyncio.Queue[Any]" = asyncio.Queue()
        self._ready = asyncio.Event()
        self._start_error: Optional[BaseException] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> "AsyncMCPClient":
        self._task = asyncio.create_task(self._run())
        try:
            # Startup includes the selected vendor's dependency installation/build.
            await asyncio.wait_for(self._ready.wait(), timeout=180)
        except BaseException:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
            raise
        if self._start_error is not None:
            raise RuntimeError("Failed to start MCP server") from self._start_error
        return self

    async def _run(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            params = StdioServerParameters(command=self._command, args=self._args, env=self._env, cwd=self._cwd)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write, elicitation_callback=self._elicitation_callback) as session:
                    await session.initialize()
                    self._ready.set()
                    while True:
                        item = await self._requests.get()
                        if item is None:
                            break
                        method, arg, fut = item
                        if fut.cancelled():
                            continue
                        try:
                            if method == "list_tools":
                                result = await asyncio.wait_for(session.list_tools(), timeout=30)
                                if not fut.done():
                                    fut.set_result(getattr(result, "tools", []) or [])
                            else:
                                # Browser location needs time for the permission prompt.
                                timeout = LOCATION_TIMEOUT_S if arg[0] == "get_user_location" else TOOL_TIMEOUT_S
                                try:
                                    result = await asyncio.wait_for(session.call_tool(*arg), timeout=timeout)
                                except asyncio.TimeoutError as exc:
                                    raise TimeoutError(f"mcp_tool_timeout_after_{timeout}s") from exc
                                if getattr(result, "isError", False):
                                    raise RuntimeError(str(self._extract_payload(result)))
                                if not fut.done():
                                    fut.set_result(self._extract_payload(result))
                        except Exception as exc:
                            if not fut.done():
                                # Do not expose the suspended session owner frame to callers.
                                fut.set_exception(exc.with_traceback(None))
        except Exception as exc:
            self._start_error = exc
            self._ready.set()
        finally:
            # Wake queued callers if the transport exits unexpectedly.
            while not self._requests.empty():
                item = self._requests.get_nowait()
                if item is not None and not item[2].done():
                    item[2].set_exception(RuntimeError("MCP session closed"))

    async def close(self) -> None:
        if self._task is None:
            return
        await self._requests.put(None)
        try:
            await asyncio.wait_for(self._task, timeout=30)
        finally:
            self._task = None

    async def list_tools(self) -> List[Any]:
        return await self._request("list_tools", None)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        return await self._request("call_tool", (name, arguments))

    async def _request(self, method: str, arg: Any) -> Any:
        if self._task is None or self._task.done():
            raise RuntimeError("MCP session is not running")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._requests.put((method, arg, fut))
        return await fut

    def _extract_payload(self, result: Any) -> Any:
        payloads: List[Any] = []
        for item in getattr(result, "content", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text = (getattr(item, "text", "") or "").strip()
                if not text:
                    continue
                try:
                    payloads.append(json.loads(text))
                except json.JSONDecodeError:
                    payloads.append(text)
            elif item_type == "json":
                payloads.append(getattr(item, "json", None))
        if not payloads:
            return None
        return payloads[0] if len(payloads) == 1 else payloads
