#!/usr/bin/env python3
"""
Baseline runner that queries an OpenAI GPT model without MCP tools.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from utils import (
    LogWriter,
    compact_trace,
    capture_console_log,
    add_common_arguments,
    run_tasks,
    select_tasks,
    write_console,
    build_messages,
    build_completion_kwargs,
    request_completion,
    build_openai_client,
    estimate_llm_cost,
    get_system_prompt,
    parse_response_content,
)

_VERBOSE = False
_TTY    = sys.stdout.isatty()
_YELLOW = "\033[33m" if _TTY else ""
_CYAN   = "\033[36m" if _TTY else ""
_GREEN  = "\033[32m" if _TTY else ""
_RESET  = "\033[0m"  if _TTY else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline LLM without MCP tools.")
    add_common_arguments(parser)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of tasks to run concurrently. 1 = sequential, with "
             "--inter-task-delay applied between calls. Above 1, tasks run with no delay.",
    )
    return parser.parse_args()


def _log(msg: str, verbose: bool = False) -> None:
    if verbose and not _VERBOSE:
        return
    write_console(msg, _TTY)


async def generate_baseline_answer(
    client,
    task: Dict[str, Any],
    model: str,
    system_prompt: str,
    idx: int = 0,
    total: int = 0,
    completion_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Step 1: Build an independent conversation for this task.
    messages = build_messages(task, system_prompt)
    task_id = task["task_id"]
    operation = task.get("operation", "?")
    _log(f"\n[{idx}/{total}][{task_id}][{operation}] --------")
    for msg in messages:
        if msg['role'] == 'system':
            _log(_YELLOW + "⚙️ System   : " + _RESET + msg['content'] + "\n")
        if msg['role'] == 'user':
            _log(_CYAN   + "👤 User     : " + _RESET + msg['content'] + "\n")

    completion_kwargs = dict(completion_kwargs or {})
    events = []

    # Step 2: Make one SDK request; its configured policy handles HTTP retries.
    start_time = time.perf_counter()
    response, error, llm_call_latency_ms = await request_completion(
        client, model, messages, completion_kwargs or {}, events)
    content = (response.choices[0].message.content or "") if error is None else None
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cached_tokens = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    if error is None:
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            error = "Response truncated (finish_reason=length) - increase max_tokens"
        elif not content.strip():
            error = "The model returned no final answer"
    _log(f"  raw_content : {response!r}", verbose=True)
    if error is None:
        messages.append({"role": "assistant", "content": content})

    # Step 3: Build the task result while preserving the full response in events.
    value, json_valid = parse_response_content(content)
    if error is not None:
        value = None
    cost_usd = getattr(usage, "cost", None)
    if cost_usd is None and response is not None:
        cost_usd = estimate_llm_cost(model, prompt_tokens, completion_tokens)
    final_latency = (time.perf_counter() - start_time) * 1000.0
    latency_breakdown = {
        "total_latency_ms": round(final_latency, 3),
        "llm_call_latency_ms": round(llm_call_latency_ms, 3),
        "llm_call_count": 1,
        "orchestration_overhead_ms": round(max(0, final_latency - llm_call_latency_ms), 3),
    }
    record = {
        "task_id": task_id,
        "status": "failed" if error else "completed",
        "output": {
            "answer": value,
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "turn_count": 1,
                "cost_usd": cost_usd,
                "latency_ms": round(final_latency, 3),
                "latency_breakdown_ms": latency_breakdown,
                "zero_hit": value is None,
                "json_valid": json_valid,
            },
        },
        "trace": compact_trace(messages, events, completion_kwargs or {}),
    }

    if error is not None:
        _log(f"  [ERROR] {task_id}  {error}")
        record["output"]["error"] = error
        return record

    try:
        display_content = json.loads(content)
        display_content = json.dumps(display_content, indent=2, ensure_ascii=False)
    except Exception:
        display_content = content
    _log(_GREEN + "🤖 Assistant: " + _RESET + display_content + "\n")
    zero_hit_flag = "  [zero_hit]" if value is None else ""
    cost_str = f"${cost_usd}" if cost_usd is not None else "?"
    _log(f"✅ Value    : {value}")
    _log(f"⏱️ Latency  : {round(final_latency, 3):.0f}ms  turns=1  tools=0")
    _log(f"🔢 Tokens   : prompt={prompt_tokens} (cached={cached_tokens})  completion={completion_tokens}  total={prompt_tokens + completion_tokens}")
    _log(f"💰 Cost     : {cost_str}{zero_hit_flag}")
    _log("-" * 62 + "\n")

    return record


async def main(args=None) -> None:
    global _VERBOSE
    # Step 1: Read CLI settings and select tasks.
    args = args or parse_args()
    _VERBOSE = args.verbose
    if args.num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    tasks = select_tasks(args)

    # Step 2: Prepare provider settings and the baseline system prompt.
    completion_kwargs = build_completion_kwargs(args)
    completion_kwargs.update(response_format={"type": "json_object"}, tool_choice="none")
    total = len(tasks)

    system_prompt = get_system_prompt("baseline", args.default_ip_location, args.system_prompt)

    async def worker(idx: int, task: Dict[str, Any]) -> Dict[str, Any]:
        return await generate_baseline_answer(
            client=client,
            task=task,
            model=args.model,
            system_prompt=system_prompt,
            idx=idx + 1,
            total=total,
            completion_kwargs=completion_kwargs,
        )


    # Step 3: The coordinator checkpoints completed tasks into one run file.
    client = build_openai_client(args.provider, max_retries=args.max_retries)
    try:
        with LogWriter(args, "baseline", tasks, system_prompt, [], completion_kwargs, endpoint=str(client.base_url)) as log_writer:
            await log_writer.warmup(tasks, worker, args.inter_task_delay)
            results = await run_tasks(tasks, worker, num_workers=args.num_workers,
                                       inter_task_delay=args.inter_task_delay, tty=_TTY,
                                       on_result=log_writer.record, on_tick=log_writer.checkpoint)
    finally:
        await client.close()
    zero_hits = sum(1 for r in results if r["output"]["metrics"]["zero_hit"])
    print(f"[baseline] Done. {len(results)}/{len(tasks)} tasks  zero_hits={zero_hits}/{len(results)}")
    print(f"[baseline] results -> {args.output}")


if __name__ == "__main__":
    args = parse_args()
    console_log_path = Path(args.terminal_log or Path(args.output).with_suffix(".log"))
    if console_log_path.resolve() == Path(args.output).resolve():
        raise ValueError("Terminal backup and run JSON must use different paths")
    with capture_console_log(console_log_path):
        asyncio.run(main(args))
