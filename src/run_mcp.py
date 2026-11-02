#!/usr/bin/env python3
"""
MCP-enabled runner that lets an OpenAI GPT model call geospatial tools via a PostGIS MCP server.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from mcp_client import AsyncMCPClient
from utils import (
    LogWriter,
    compact_trace,
    capture_console_log,
    agent_notice,
    build_tool_schema,
    add_common_arguments,
    run_tasks,
    select_tasks,
    write_console,
    build_messages,
    build_completion_kwargs,
    request_completion,
    reasoning_of,
    NO_JSON_MODE_WITH_TOOLS,
    build_openai_client,
    default_location_elicitation,
    estimate_llm_cost,
    get_system_prompt,
    parse_response_content,
)

_VERBOSE = False
_TTY = sys.stdout.isatty()
_YELLOW   = "\033[33m" if _TTY else ""
_CYAN     = "\033[36m" if _TTY else ""
_GREEN    = "\033[32m" if _TTY else ""
_MAGENTA  = "\033[35m" if _TTY else ""
_RESET    = "\033[0m"  if _TTY else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM with spatial MCP tools.",
        allow_abbrev=False,
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--mcp-command",
        default="make",
        help="Command to launch the OSM MCP server over stdio.",
    )
    parser.add_argument(
        "--mcp-arg",
        action="append",
        default=None,
        help="Repeatable argument to pass to the MCP server command.",
    )
    parser.add_argument(
        "--mcp-env",
        action="append",
        default=[],
        help="Repeatable KEY=VALUE pair added to the MCP server environment.",
    )
    parser.add_argument(
        "--mcp-vendor",
        default="purduedb/limo",
        help="The MCP vendor to use.",
    )
    parser.add_argument(
        "--mcp-tools",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated tool names, overrides --mcp-vendor's tool allowlist.",
    )
    parser.add_argument(
        "--knn-limit",
        type=int,
        default=None,
        help="Override the number of KNN results requested from tools (e.g., 10 to inspect deeper ranks).",
    )
    parser.add_argument(
        "--max-agent-turns",
        type=int,
        default=20,
        help="Model-call budget per task, including the final answer-only call (default: 20).",
    )
    args = parser.parse_args()
    if args.mcp_arg is None:
        if args.mcp_command == "make":
            args.mcp_arg = ["--silent", "--no-print-directory", "mcp_serve", f"MCP_VENDOR={args.mcp_vendor}"]
        else:
            args.mcp_arg = []
    return args


def _log(msg: str, verbose: bool = False) -> None:
    if verbose and not _VERBOSE:
        return
    write_console(msg, _TTY)


def parse_env_vars(items: List[str]) -> Dict[str, str]:
    env = dict(os.environ)
    for item in items:
        if "=" not in item:
            raise ValueError("Invalid --mcp-env entry. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        env[key] = value
    return env


async def execute_tool_call(
    client: AsyncMCPClient,
    name: str,
    arguments: Dict[str, Any],
    known_tools: Optional[Iterable[str]] = None,
    task: Optional[Dict[str, Any]] = None,
    knn_limit: Optional[int] = None,
) -> Any:
    # Preserve the CLI override used to inspect deeper KNN ranks.
    if knn_limit and task and task.get("operation") in {"knn_query", "knn_query_network"}:
        if name == "find_nearby_places":
            arguments["limit"] = max(int(arguments.get("limit", 0)), knn_limit)
        if name in {"knn-geodesic", "knn-driving"}:
            arguments["k"] = max(int(arguments.get("k", 0)), knn_limit)

    # Reject unknown tools before dispatch.
    if known_tools is None or name in known_tools:
        return await client.call_tool(name, arguments)
    raise RuntimeError(f"Unsupported tool: {name}")


async def agent_loop(
    openai_client,
    spatial_client: AsyncMCPClient,
    task: Dict[str, Any],
    model: str,
    system_prompt: str,
    tools: List[Dict[str, Any]],
    known_tools: Optional[Iterable[str]],
    knn_limit: Optional[int],
    max_agent_turns: int,
    completion_kwargs: Optional[Dict[str, Any]] = None,
    idx: int = 0,
    total: int = 0,
) -> Dict[str, Any]:
    """Run tool turns; the SDK handles HTTP retries inside each model request."""
    if not isinstance(max_agent_turns, int) or isinstance(max_agent_turns, bool) or max_agent_turns < 1:
        raise ValueError("max_agent_turns must be a positive integer")
    # Step 1: Build the task conversation and initialize accounting.
    messages = build_messages(task, system_prompt)
    task_id = task["task_id"]
    operation = task.get("operation", "?")
    _log(f"\n[{idx}/{total}][{task_id}][{operation}] --------")
    for msg in messages:
        if msg['role'] == 'system':
            _log(_YELLOW + "⚙️ System   : " + _RESET + msg['content'] + "\n")
        if msg['role'] == 'user':
            _log(_CYAN   + "👤 User     : " + _RESET + msg['content'] + "\n")
    events = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_cost: Optional[float] = None  # Provider-reported, cache-aware
    cost_complete = True
    start_time = time.perf_counter()
    llm_call_latency_ms = 0.0
    llm_call_count = 0
    tool_call_latency_ms = 0.0
    tool_call_count = 0
    timeout_tool_latency_ms = 0.0
    timeout_tool_count = 0

    def build_latency_breakdown(total_latency_ms: float) -> Dict[str, Any]:
        orchestration_overhead_ms = max(
            0.0,
            total_latency_ms - llm_call_latency_ms - tool_call_latency_ms,
        )
        latency_breakdown = {
            "total_latency_ms": round(total_latency_ms, 3),
            "llm_call_latency_ms": round(llm_call_latency_ms, 3),
            "llm_call_count": llm_call_count,
            "tool_call_latency_ms": round(tool_call_latency_ms, 3),
            "tool_call_count": tool_call_count,
            "timeout_tool_latency_ms": round(timeout_tool_latency_ms, 3),
            "timeout_tool_count": timeout_tool_count,
            "orchestration_overhead_ms": round(orchestration_overhead_ms, 3),
        }
        _log(f"⏱️  Latency breakdown (ms): {latency_breakdown}\n", verbose=True)
        return latency_breakdown

    call_counts = {}
    repeated_calls = 0
    failure_error = None
    final_content = None
    turn_count = 0
    while turn_count < max_agent_turns:
        turn_count += 1
        # Step 2: Append the budget and repeated-call guidance once per turn.
        messages.append(agent_notice(turn_count, max_agent_turns, repeated_calls))
        turn_kwargs = dict(completion_kwargs or {})
        if turn_count == max_agent_turns and tools:
            turn_kwargs["tool_choice"] = "none"
        _tool_names_str = ", ".join(t["function"]["name"] for t in tools)
        _log(
            f"  {_CYAN}⬆️  API req  :{_RESET} turn={turn_count}/{max_agent_turns}  model={model}\n"
            f"      messages    : {len(messages)} sent\n"
            f"      temperature : {(completion_kwargs or {}).get('temperature', '?')}"
            f"  seed={(completion_kwargs or {}).get('seed', '?')}"
            f"  max_tokens={(completion_kwargs or {}).get('max_tokens', '?')}"
            f"  reasoning={reasoning_of(completion_kwargs) or 'default'}\n"
            f"      tool_choice : {turn_kwargs.get('tool_choice', '?')}\n"
            f"      tools       : {len(tools)}  [{_tool_names_str}]\n",
            verbose=True,
        )
        # Step 3: The SDK handles transport retries without advancing agent turns.
        parameters = dict(turn_kwargs, tools=tools)
        if not any(family in model.lower() for family in NO_JSON_MODE_WITH_TOOLS):
            parameters["response_format"] = {"type": "json_object"}
        response, llm_error, _turn_latency_ms = await request_completion(
            openai_client, model, messages, parameters, events, turn_count)
        llm_call_count += 1
        llm_call_latency_ms += _turn_latency_ms
        if llm_error is not None:
            failure_error = f"LLM request failed after SDK retries: {llm_error}"
            break

        # Step 4: Record the response and accumulate usage.
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        cached_tokens = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        total_cached_tokens += cached_tokens
        _turn_cost = getattr(usage, "cost", None)
        if _turn_cost is not None:
            total_cost = (total_cost or 0) + _turn_cost
        else:
            cost_complete = False
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        finish_reason = response.choices[0].finish_reason
        _resp_model = getattr(response, "model", model)
        _tc_names = "  [" + ", ".join(tc.function.name for tc in tool_calls) + "]"
        _log(
            f"  {_GREEN}⬇️  API resp :{_RESET} turn={turn_count}/{max_agent_turns}  model={_resp_model}  finish_reason={finish_reason} ({_turn_latency_ms:.0f}ms)\n"
            f"      tokens      : prompt={prompt_tokens} (cached={cached_tokens})  completion={completion_tokens}  total={prompt_tokens + completion_tokens}\n"
            f"      cost        : {_turn_cost if _turn_cost is not None else estimate_llm_cost(_resp_model, prompt_tokens, completion_tokens) or '?'}\n"
            f"      tool_calls  : {len(tool_calls)}{_tc_names}\n"
            f"      raw_content : {response!r}\n",
            verbose=True,
        )

        # Append assistant message with tool calls (if any) for next turn
        messages.append({
            "role": "assistant",
            "content": message.content,
            **({"tool_calls": [tc.model_dump() for tc in message.tool_calls]} if message.tool_calls else {}),
        })
        _log(_GREEN + "🤖 Assistant: " + _RESET + f"Message({message.content}), Tool Calls({json.dumps([tc.model_dump() for tc in message.tool_calls] if message.tool_calls else [])})" + "\n")

        # Step 5: Execute requested tools and append their results for the next call.
        if finish_reason == "tool_calls":
            if turn_count == max_agent_turns:
                failure_error = f"Agent turn limit reached ({max_agent_turns}); the model requested tools on the final answer-only call"
                break
            repeated_calls = 0
            for tool_call in message.tool_calls:
                args_text = tool_call.function.arguments or "{}"
                invalid_arguments = False
                try:
                    arguments = json.loads(args_text)
                except json.JSONDecodeError:
                    arguments = {}
                    invalid_arguments = True
                requested_arguments = copy.deepcopy(arguments)
                key = (tool_call.function.name, json.dumps(arguments, sort_keys=True, separators=(",", ":")))
                call_counts[key] = call_counts.get(key, 0) + 1
                if call_counts[key] > 1:
                    repeated_calls += 1
                tool_start = time.perf_counter()
                error: Optional[str] = None
                timeout_exception = False
                try:
                    tool_result = await execute_tool_call(
                        spatial_client,
                        tool_call.function.name,
                        arguments,
                        known_tools,
                        task,
                        knn_limit,
                    )
                except Exception as exc:  # pragma: no cover - network/tool errors
                    timeout_exception = isinstance(exc, TimeoutError)
                    error = str(exc)
                    tool_result = {"error": error}
                    _log(f"  [ERROR] {task_id}  tool_error [{tool_call.function.name}]: {error}")
                tool_latency = (time.perf_counter() - tool_start) * 1000.0
                tool_call_count += 1
                tool_call_latency_ms += tool_latency
                _log(
                    f"{_MAGENTA}🔧 Tool     :{_RESET} [{tool_call_count}]  {tool_call.function.name}  ({tool_latency:.0f}ms)\n"
                    f"  input     : {json.dumps(arguments)}\n"
                    f"  output    : {json.dumps(tool_result)}\n"
                )
                timeout_hit = timeout_exception or ("timeout" in (error or "").lower()) or (
                    "timed out" in (error or "").lower()
                )
                if timeout_hit:
                    timeout_tool_count += 1
                    timeout_tool_latency_ms += tool_latency
                event = {"type": "tool_call", "sequence": len(events) + 1, "turn": turn_count,
                         "call_id": tool_call.id,
                         "started_at": time.time() - tool_latency / 1000, "finished_at": time.time(),
                         "latency_ms": round(tool_latency, 3), "timeout": bool(timeout_hit),
                         "request_transcript_index": next(i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "assistant"),
                         "result_transcript_index": len(messages)}
                if error:
                    event["error"] = error
                # Keep effective arguments when a CLI override changes the model's request.
                if invalid_arguments or arguments != requested_arguments:
                    event["arguments"] = arguments
                events.append(event)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False, separators=(",", ":")),
                })
            # Continue loop for tool outputs
            continue

        # Reject incomplete final answers.
        if finish_reason == "length":
            final_content = message.content
            failure_error = "Response truncated (finish_reason=length) - increase max_tokens"
            break

        if not (message.content or "").strip():
            final_content = message.content or ""
            failure_error = "The agent returned no final answer" + (f" within the {max_agent_turns}-turn limit" if turn_count == max_agent_turns else "")
            break

        # Log the final answer once, then finalize the task.
        try:
            display_content = json.loads(message.content or "")
            display_content = json.dumps(display_content, indent=2, ensure_ascii=False)
        except Exception:
            display_content = message.content or ""
        _log(_GREEN + "🤖 Assistant: " + _RESET + display_content + "\n")

        final_content = message.content
        break

    # Step 6: Build the final saved task record.
    if final_content is None:
        failure_error = failure_error or f"max_agent_turns_exceeded:{max_agent_turns}"
        _log(f"  [ERROR] {task_id}  {failure_error}")
    final_latency = (time.perf_counter() - start_time) * 1000.0
    latency_breakdown = build_latency_breakdown(final_latency)
    cost_usd = total_cost if cost_complete and total_cost is not None else estimate_llm_cost(model, total_prompt_tokens, total_completion_tokens)
    value, json_valid = parse_response_content(final_content)
    if failure_error is not None:
        value = None
    record = {
        "task_id": task_id,
        "status": "failed" if failure_error else "completed",
        "output": {
            "answer": value,
            "metrics": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "cached_tokens": total_cached_tokens,
                "turn_count": turn_count,
                "cost_usd": cost_usd,
                "latency_ms": round(final_latency, 3),
                "latency_breakdown_ms": latency_breakdown,
                "zero_hit": value is None,
                "json_valid": json_valid,
            },
        },
        "trace": compact_trace(messages, events, completion_kwargs or {}),
    }
    if failure_error is not None:
        record["output"]["error"] = failure_error
    zero_hit_flag = "  [zero_hit]" if value is None else ""
    cost_str = f"${cost_usd}" if cost_usd is not None else "?"
    _log(f"✅ Value    : {value}")
    _log(f"⏱️ Latency  : {round(final_latency, 3):.0f}ms  turns={turn_count}  tools={tool_call_count}")
    _log(f"🔢 Tokens   : prompt={total_prompt_tokens} (cached={total_cached_tokens})  completion={total_completion_tokens}  total={total_prompt_tokens + total_completion_tokens}")
    _log(f"💰 Cost     : {cost_str}{zero_hit_flag}")
    _log("-" * 62 + "\n")
    return record


async def main(args=None) -> None:
    global _VERBOSE
    # Step 1: Read CLI settings and validate the task selection.
    args = args or parse_args()
    _VERBOSE = args.verbose
    tasks = select_tasks(args)
    mcp_env = parse_env_vars(args.mcp_env)

    # Step 2: Start the shared MCP session and select its exposed tools.
    openai_client = build_openai_client(args.provider, max_retries=args.max_retries)
    print(f"[mcp] Starting MCP server: {args.mcp_command} {' '.join(args.mcp_arg or [])}")
    mcp_client = AsyncMCPClient(
        command=args.mcp_command,
        args=args.mcp_arg,
        env=mcp_env,
        # Runners have no device: location requests get the fixed default position (reproducible).
        elicitation_callback=default_location_elicitation,
    )
    try:
        await mcp_client.start()
        # Expose all live tools unless the CLI selects a subset.
        available_tools = await mcp_client.list_tools()
        if args.mcp_tools:
            tools_to_include = set(args.mcp_tools)
            available_tools = [t for t in available_tools if t.name in tools_to_include]
        tool_schema = build_tool_schema(available_tools)
        tool_names: List[str] = [t.name for t in available_tools if t.name]
        known_tools = tool_names or None
        print(f"[mcp] MCP tools ({len(tool_names)}): ")
        for name in tool_names:
            print(f"  - {name}")

        # Step 3: Prepare tasks, request settings, and the system prompt.
        completion_kwargs = build_completion_kwargs(args)
        completion_kwargs["tool_choice"] = "auto"

        total = len(tasks)
        system_prompt = get_system_prompt("mcp", args.default_ip_location, args.system_prompt)

        async def worker(idx: int, task: Dict[str, Any]) -> Dict[str, Any]:
            return await agent_loop(
                openai_client=openai_client,
                spatial_client=mcp_client,
                task=task,
                model=args.model,
                system_prompt=system_prompt,
                tools=tool_schema,
                known_tools=known_tools,
                knn_limit=args.knn_limit,
                max_agent_turns=args.max_agent_turns,
                completion_kwargs=completion_kwargs,
                idx=idx + 1,
                total=total,
            )

        # Step 4: One coordinator owns checkpoints; MCP keeps one worker.
        with LogWriter(args, "mcp", tasks, system_prompt, tool_schema, completion_kwargs, endpoint=str(openai_client.base_url)) as log_writer:
            await log_writer.warmup(tasks, worker, args.inter_task_delay)
            results = await run_tasks(tasks, worker, inter_task_delay=args.inter_task_delay, tty=_TTY,
                                       on_result=log_writer.record, on_tick=log_writer.checkpoint)
    finally:
        try:
            await mcp_client.close()
        finally:
            await openai_client.close()

    zero_hits = sum(1 for r in results if r["output"]["metrics"]["zero_hit"])
    print(f"[mcp] Done. {len(results)}/{len(tasks)} tasks  zero_hits={zero_hits}/{len(results)}")
    print(f"[mcp] results -> {args.output}")


if __name__ == "__main__":
    args = parse_args()
    console_log_path = Path(args.terminal_log or Path(args.output).with_suffix(".log"))
    if console_log_path.resolve() == Path(args.output).resolve():
        raise ValueError("Terminal backup and run JSON must use different paths")
    with capture_console_log(console_log_path):
        asyncio.run(main(args))
