#!/usr/bin/env python3
"""Shared runner setup, requests, orchestration, prompts, and logging."""

from __future__ import annotations

import json
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import os
import re
import sys
import threading
import tempfile
import traceback
import uuid
import asyncio
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import litellm
from openai import AsyncOpenAI, OpenAIError
from dotenv import load_dotenv
from tqdm import tqdm

from mcp_client import TOOL_TIMEOUT_S, LOCATION_TIMEOUT_S

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "genai": {
        # Reference: https://www.rcac.purdue.edu/knowledge/genaistudio?all=true
        "base_url": "https://genai.rcac.purdue.edu/api",
        "api_key_env": "GENAI_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
}

SYSTEM_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "unified"
SYSTEM_PROMPT_FILES = {"baseline": "baseline.txt", "mcp": "tools.txt"}

_CODE_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# JSON mode can disrupt tool calling; these exclusions apply to MCP, not baseline.
NO_JSON_MODE_WITH_TOOLS = ("gemini", "gemma", "deepseek", "qwen", "z-ai/glm-5.3")

_PROVIDER_REMAP = {
    "qwen/": "dashscope/",
}

_OPENROUTER_CATALOG = os.path.join(
    os.path.dirname(__file__), "../data/openrouter_models_raw_20260723.json"
)


# Run setup and task coordination.


def add_common_arguments(parser) -> None:
    """Register shared CLI inputs; each runner adds its own execution options."""
    # input / output
    parser.add_argument("--tasks", action="append", required=True,
        help="Path(s) to JSON task files (list or {'tasks': [...]})")
    parser.add_argument("--output", required=True,
        help="Single run JSON file (for example outputs/run.json).")
    parser.add_argument("--terminal-log", default=None,
        help="Console log path (default: output path with .log suffix).")

    # task selection
    parser.add_argument("--operation", type=str, default=None,
        help="Filter tasks to only this operation type (e.g. knn, geocode).")
    parser.add_argument("--task-id", dest="task_ids", nargs="+", default=None, metavar="TASK_ID",
        help="Run only the specified task ID(s), space-separated (e.g. WL-PATH-001 WL-KNN-002).")
    parser.add_argument("--limit", type=int, default=None,
        help="Cap the number of tasks to run (takes the first N after loading).")

    # model
    parser.add_argument("--provider", default="openrouter", choices=list(PROVIDERS),
        help="API provider (openai, gemini, openrouter, genai, anthropic).")
    parser.add_argument("--model", default="openai/gpt-6-luna", help="Provider model identifier.")
    parser.add_argument("--reasoning", default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        help="Provider reasoning effort override (e.g. 'none' to disable reasoning tokens "
            "entirely). Unset by default, i.e. the model/provider's own default applies.")
    parser.add_argument("--temperature", type=float, default=0,
        help="LLM sampling temperature (default: 0 for deterministic output).")
    parser.add_argument("--max-tokens", type=int, default=4096,
        help="Maximum tokens in LLM response (default: 4096).")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducible LLM responses (default: 42).")

    # prompt
    parser.add_argument("--system-prompt", help="Override the condition's rendered system prompt with a file.")
    parser.add_argument("--default-ip-location", action="store_true",
        help="Use the fixed prompts/unified/default_ip_location.json in <ip_location> instead of a live lookup (reproducible).")

    # execution
    parser.add_argument("--max-retries", type=int, default=3,
        help="SDK HTTP retries per request, in addition to the initial attempt.")
    parser.add_argument("--inter-task-delay", type=float, default=0.0,
        help="Optional delay (seconds) between tasks to smooth TPM usage.")
    parser.add_argument("--warmup-runs", type=int, default=1,
        help="Repeat the first selected task before measurement (default: 1; 0 disables).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")


def select_tasks(args) -> List[Dict[str, Any]]:
    """Apply filters in CLI order and validate tasks before starting workers."""
    if getattr(args, "warmup_runs", 1) < 0:
        raise ValueError("--warmup-runs must be 0 or greater")
    tasks = load_tasks(args.tasks)
    if args.operation is not None:
        tasks = [task for task in tasks if task.get("operation") == args.operation]
    if args.task_ids is not None:
        wanted = set(args.task_ids)
        tasks = [task for task in tasks if task.get("task_id") in wanted]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    if any("task_id" not in task for task in tasks):
        raise ValueError("Every task must include a 'task_id'.")
    return tasks

async def run_tasks(tasks: list, worker: Callable, *, num_workers: int = 1,
                    inter_task_delay: float = 0, tty: bool = False,
                    on_result: Optional[Callable] = None, on_tick: Optional[Callable] = None) -> list:
    """Bound task concurrency; only the coordinator writes completed records."""
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    results = [None] * len(tasks)
    next_index = 0
    pending = {}

    async def execute(index):
        # Delays apply only to sequential runs and stay inside the checkpoint heartbeat.
        if num_workers == 1 and index and inter_task_delay > 0:
            await asyncio.sleep(inter_task_delay)
        return await worker(index, tasks[index])

    with tqdm(total=len(tasks), disable=not tty) as progress:
        try:
            while next_index < len(tasks) or pending:
                while next_index < len(tasks) and len(pending) < num_workers:
                    pending[asyncio.create_task(execute(next_index))] = next_index
                    next_index += 1
                done, _ = await asyncio.wait(pending, timeout=1, return_when=asyncio.FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: pending[f]):
                    index = pending.pop(future)
                    record = future.result()
                    results[index] = record
                    if on_result:
                        on_result(index, record)
                    progress.update()
                if on_tick:
                    on_tick()
        finally:
            # Finish cancellation before clients close their transports.
            for future in pending:
                future.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    return results


# Provider configuration and requests.


def build_openai_client(
    provider: str = "openrouter",
    api_key_env: Optional[str] = None,
    max_retries: int = 2,
) -> AsyncOpenAI:
    """
    Create the shared async OpenAI-compatible client.
    """
    config = PROVIDERS.get(provider)
    if config is None:
        known = ", ".join(PROVIDERS)
        raise RuntimeError(
            f"Unknown provider '{provider}'. Known providers: {known}"
        )

    key_env = api_key_env or config["api_key_env"]
    api_key = os.getenv(key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider '{provider}'. "
            f"Set the `{key_env}` environment variable (or add it to your .env file)."
        )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.environ.get(config["api_key_env"].removesuffix("_API_KEY") + "_BASE_URL") or config["base_url"],
        max_retries=max_retries,
    )
    return client

def build_completion_kwargs(args) -> Dict[str, Any]:
    """Shared sampling, reasoning, and provider usage parameters."""
    kwargs = {key: getattr(args, key) for key in ("temperature", "seed", "max_tokens")
              if getattr(args, key) is not None}
    kwargs.update(provider_parameters(args.provider, args.reasoning))
    return kwargs

def provider_parameters(provider: str, reasoning: Optional[str]) -> dict:
    """OpenRouter extends the API; other providers use the SDK reasoning field."""
    if provider == "openrouter":
        extra = {"usage": {"include": True}}
        if reasoning is not None:
            extra["reasoning"] = {"effort": reasoning}
        return {"extra_body": extra}
    return {"reasoning_effort": reasoning} if reasoning is not None else {}


def reasoning_of(kwargs: Optional[dict]) -> Optional[str]:
    kwargs = kwargs or {}
    return kwargs.get("reasoning_effort") or (kwargs.get("extra_body") or {}).get("reasoning", {}).get("effort")

def cache_system(messages: list, client: AsyncOpenAI) -> list:
    """Mark only the initial system prompt cacheable on OpenRouter, not runtime notices."""
    if "openrouter" not in str(client.base_url):
        return messages
    return [
        {**m, "content": [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]} if i == 0 and m["role"] == "system" else m
        for i, m in enumerate(messages)
    ]

def build_completion_request(client, model, messages, parameters):
    """Build the effective SDK request shared by runners and demo streaming calls."""
    request = dict(model=model, messages=cache_system(messages, client), **parameters)
    if str(client.base_url).rstrip('/') == "https://api.anthropic.com/v1":
        if request.get("response_format", {}).get("type") == "json_object":
            request.pop("response_format")  # Direct endpoint rejects JSON-object mode.
        request.pop("seed", None)  # Ignored by Anthropic; optional cleanup.
        request.pop("temperature", None)  # Newer Claude models reject values below 1.
        if not request.get("tools"):
            request.pop("tool_choice", None)  # No tools to select; removal may be unnecessary.
        effort = request.pop("reasoning_effort", None)  # Ignored as-is; translate below.
        if effort is not None:
            if effort == "minimal":
                raise ValueError("Anthropic does not support minimal effort; use low.")
            extra = dict(request.get("extra_body") or {})
            extra.update({"thinking": {"type": "disabled"}} if effort == "none" else
                         {"output_config": {"effort": effort}})
            request["extra_body"] = extra
    return request


async def request_completion(client, model, messages, parameters, events, turn=1):
    """The SDK owns HTTP retries; record its final response and total elapsed time."""
    request = build_completion_request(client, model, messages, parameters)
    started_at = time.time()
    start = time.perf_counter()
    response, error = None, None
    try:
        response = await client.chat.completions.create(**request)
        error = completion_response_error(response)
    except OpenAIError as exc:
        error = str(exc)
    elapsed = (time.perf_counter() - start) * 1000
    events.append({
        "type": "llm_call", "sequence": len(events) + 1, "turn": turn,
        "started_at": started_at, "finished_at": time.time(), "latency_ms": round(elapsed, 3),
        "request": copy.deepcopy(request),
        "response": response.model_dump(mode="json") if response is not None else None,
        "error": error,
    })
    return response, error, elapsed


def completion_response_error(response: Any) -> Optional[str]:
    """Validate API envelopes before either runner indexes choices."""
    if response is None:
        return f"reason=null response\n      raw_content : {response!r}"
    if not response.choices:
        return f"reason=empty choices\n      raw_content : {response!r}"
    error = getattr(response, "error", None)
    if error is not None:
        return f"reason=API error {error}\n      raw_content : {response!r}"
    return None




# Agent notices, tool schemas, and prompts.


def agent_notice(turn: int, max_turns: int, repeated_calls: int = 0) -> dict:
    """Append-only runtime guidance shared by the demo and MCP runner."""
    budget = f"Turn {turn}/{max_turns}."
    if turn == 1:
        budget = "New query. Earlier budget and repeated-call notices no longer apply. " + budget
    if turn == max_turns:
        budget += (" Final call. Tools disabled. Return the required JSON using verified results only. "
                   "State unresolved requirements in notes; do not present partial results as complete.")
    elif turn == max_turns - 1:
        budget += (" 2 calls left, including this call. Make only essential tool calls; "
                   "the next call must answer without tools.")
    elif turn == max_turns - 4:
        budget += " 5 calls left, including this call. Focus on unresolved requirements and prepare the final answer."
    guidance = f"<agent_budget>\n{budget}\n</agent_budget>"
    if repeated_calls:
        guidance += (f"\n<repeated_tool_calls>\nLast batch repeated {repeated_calls} tool call(s) with identical arguments. "
                     "Reuse successful results. For persistent errors, fix arguments or change strategy. "
                     "Retry unchanged only for transient failures.\n</repeated_tool_calls>")
    return {"role": "system", "content": f"<system-reminder>\n{guidance}\n</system-reminder>"}


def build_tool_schema(tools: Iterable[Any]) -> List[Dict[str, Any]]:
    schema: List[Dict[str, Any]] = []
    for tool in tools:
        if not tool.name:
            continue
        # mcp SDK v2 renamed Tool.inputSchema -> input_schema (snake_case); support both.
        input_schema = copy.deepcopy(getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {})
        input_schema.setdefault("type", "object")
        input_schema.setdefault("properties", {})
        schema.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": input_schema,
                },
            }
        )
    return schema

def get_system_prompt(pane: str, default_ip_location: bool = False, path: Optional[str] = None) -> str:
    """Rendered unified prompt for a pane ("baseline" or "mcp") plus the <ip_location> block."""
    if path is not None:
        return Path(path).read_text(encoding="utf-8").strip()
    text = (SYSTEM_PROMPTS_DIR / "rendered" / SYSTEM_PROMPT_FILES[pane]).read_text(encoding="utf-8").strip()
    return f"{text}\n\n<ip_location>\n{json.dumps(get_ip_location(default_ip_location), indent=2)}\n</ip_location>"

def build_messages(task: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.get("prompt", "")},
    ]

def get_ip_location(default_only: bool = False) -> Dict[str, Any]:
    """IP location from ip-api.com; the default file is the fallback on any failure or when default_only."""
    default = json.loads((SYSTEM_PROMPTS_DIR / "default_ip_location.json").read_text(encoding="utf-8"))
    if default_only:
        # Return the default location without making an external request (for reproducibile benchmarks).
        return default
    try:
        # Live request to ip-api.com (for simulating production behavior, may not be reproducible).
        with urllib.request.urlopen(f"http://ip-api.com/json/?fields={','.join(default)}", timeout=3) as resp:
            data = json.load(resp)
        if data.get("status") == "success":
            return {k: data.get(k, default[k]) for k in default}
    except Exception:
        pass
    # Return the default location if the live request fails or is not successful.
    return default

def get_default_user_location() -> Dict[str, Any]:
    """Fixed user position for runs without a device (reproducible)."""
    return json.loads((SYSTEM_PROMPTS_DIR / "default_user_location.json").read_text(encoding="utf-8"))

async def default_location_elicitation(context: Any, params: Any) -> Any:
    """Elicitation handler for clients without a device: answers with the fixed default position."""
    from mcp.types import ElicitResult

    return ElicitResult(action="accept", content=get_default_user_location())


# Response parsing and cost estimation.


def parse_response_content(content: Optional[str]) -> Tuple[Any, Optional[bool]]:
    """Return the recovered JSON value and whether the original text is valid JSON.

    Missing output has no validity result. Fenced/prose-wrapped JSON may be
    recovered, but arbitrary text is never treated as a usable answer.
    """
    if content is None:
        return None, None

    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    try:
        return json.loads(content, parse_constant=reject_constant), True
    except ValueError:
        candidate = _strip_code_fence(content)
        for text in (candidate, _extract_json_object(candidate)):
            if text:
                try:
                    return json.loads(text, parse_constant=reject_constant), False
                except ValueError:
                    pass
        return None, False


def estimate_llm_cost(
    model: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]
) -> Optional[float]:
    if prompt_tokens is None or completion_tokens is None:
        return None

    for src, dst in _PROVIDER_REMAP.items():
        if model.startswith(src):
            model = dst + model[len(src):]
            break

    try:
        info = litellm.get_model_info(model)
        # Fall back to the catalog when LiteLLM has no usable price.
        if not info.get("input_cost_per_token") and not info.get("output_cost_per_token"):
            info = _openrouter_price(model)
    except Exception:
        info = _openrouter_price(model)

    if not info:
        return None
    cost = prompt_tokens * info["input_cost_per_token"] + completion_tokens * info["output_cost_per_token"]
    return round(cost, 6)

def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()

def _extract_json_object(text: str) -> Optional[str]:
    match = _JSON_OBJECT_RE.search(text)
    if match:
        return match.group(0)
    return None

def _openrouter_price(model: str) -> Optional[Dict[str, float]]:
    model = model.removeprefix("openrouter/")
    with open(_OPENROUTER_CATALOG) as fp:
        catalog = {m["id"]: m["pricing"] for m in json.load(fp)["data"]}
    pricing = catalog.get(model)
    if not pricing:
        return None
    return {"input_cost_per_token": float(pricing["prompt"]), "output_cost_per_token": float(pricing["completion"])}


# File and console output.


def load_tasks(paths: Iterable[str]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict) and "tasks" in data:
            task_list = data["tasks"]
        else:
            task_list = data
        if not isinstance(task_list, list):
            raise ValueError(f"{path} must contain a list of tasks.")
        tasks.extend(task_list)
    return tasks


def write_console(message: str, tty: bool) -> None:
    """Write one message without yielding between async workers."""
    if tty:
        tqdm.write(str(message))
    else:
        print(message, flush=True)


# Run JSON checkpoints and console logs.


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def compact_trace(messages, events, completion_kwargs):
    """Reference shared request payloads without duplicating them in the trace."""
    # Keep a readable transcript and index each request into its immutable prefix.
    transcript = copy.deepcopy(messages)
    if transcript and transcript[0].get("role") == "system":
        transcript[0]["content"] = "SYSTEM_PROMPT"
    events = copy.deepcopy(events)
    for event in events:
        if event["type"] == "llm_call":
            request = event["request"]
            parameters = {key: value for key, value in request.items() if key not in {"model", "messages", "tools"}}
            event["request"] = {
                "message_count": len(request["messages"]),
                "overrides": {key: value for key, value in parameters.items() if key not in completion_kwargs or value != completion_kwargs[key]},
            }
            if "tools" in request:
                event["request"]["tools"] = "TOOLS"
            # Preserve the provider-specific cache annotation without duplicating messages.
            if request["messages"] and isinstance(request["messages"][0].get("content"), list):
                event["request"]["cache_system"] = True
            if event.get("error") is None:
                event.pop("error", None)
    return {"transcript": transcript, "events": events}


class LogWriter:
    """Only the coordinator mutates and checkpoints the run; workers return records."""
    def __init__(self, args, condition, tasks, system_prompt, tools, completion_kwargs, endpoint=None):
        self.path = Path(args.output)
        if self.path.suffix != '.json':
            raise ValueError('--output must name a .json run file (for example run.json)')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Log only known CLI settings; never copy arbitrary environment values.
        shared = ("provider", "model", "reasoning", "tasks", "task_ids", "operation", "limit",
                  "default_ip_location", "max_retries", "inter_task_delay", "verbose")
        config = {key: copy.deepcopy(getattr(args, key)) for key in shared if getattr(args, key, None) is not None}
        config["warmup_runs"] = getattr(args, "warmup_runs", 1)
        config["tasks"] = [str(Path(path).resolve()) for path in args.tasks]
        config.update(condition=condition, endpoint=endpoint,
                      num_workers=getattr(args, "num_workers", 1),
                      completion_kwargs=copy.deepcopy(completion_kwargs))
        if condition == "mcp":
            config["num_workers"] = 1
            config["mcp"] = {
                "vendor": args.mcp_vendor, "command": args.mcp_command,
                "args": args.mcp_arg,
                "env_keys": [item.split("=", 1)[0] for item in args.mcp_env],
                "max_agent_turns": args.max_agent_turns, "tool_timeout_s": TOOL_TIMEOUT_S, "location_timeout_s": LOCATION_TIMEOUT_S,
            }
            for key in ("mcp_tools", "knn_limit"):
                if getattr(args, key, None) is not None:
                    config["mcp"][key.removeprefix("mcp_")] = getattr(args, key)
        prompt_path = getattr(args, "system_prompt", None) or str(SYSTEM_PROMPTS_DIR / "rendered" / SYSTEM_PROMPT_FILES[condition])
        self.document = {
            "run": {"id": str(uuid.uuid4()), "started_at": None, "finished_at": None,
                    "status": "running", "completed_tasks": 0, "total_tasks": len(tasks),
                    "terminal_log": str(args.terminal_log or self.path.with_suffix(".log"))},
            "config": config,
            "system_prompt": {"path": str(prompt_path), "text": system_prompt},
            "tools": {"schemas": copy.deepcopy(tools)},
            "tasks": [{"task_id": task["task_id"],
                       "input": {key: copy.deepcopy(value) for key, value in task.items() if key not in {"task_id", "answer"}},
                       "status": "pending"} for task in tasks],
        }
        self.document["warmup_tasks"] = [copy.deepcopy(self.document["tasks"][0])
                                          for _ in range(config["warmup_runs"]) if tasks]
        self.dirty = True
        self.since_checkpoint = 0
        self.last_checkpoint = time.monotonic()

    def __enter__(self):
        print(f"[{self.document['config']['condition']}] run config")
        print(json.dumps(self.document["config"], indent=2, ensure_ascii=False), flush=True)
        self.checkpoint(force=True)
        return self

    async def warmup(self, tasks, worker, inter_task_delay=0):
        """Use the normal worker with fresh conversations; retain unscored records."""
        count = len(self.document["warmup_tasks"])
        if count:
            print(f"[warmup] Running first selected task {count} time(s); excluded from evaluation.")
            await run_tasks([copy.deepcopy(tasks[0]) for _ in range(count)], worker,
                            inter_task_delay=inter_task_delay,
                            on_result=lambda i, r: self.record(i, r, warmup=True),
                            on_tick=self.checkpoint)
            if inter_task_delay > 0:
                await asyncio.sleep(inter_task_delay)
        self.document["run"]["started_at"] = utc_now()
        self.dirty = True
        self.checkpoint(force=True)
        print("[benchmark] Starting measured tasks.")

    def record(self, index, record, *, warmup=False):
        self.document['warmup_tasks' if warmup else 'tasks'][index].update(record)
        if not warmup:
            self.document['run']['completed_tasks'] += 1
        self.since_checkpoint += 1
        self.dirty = True
        self.checkpoint(force=warmup or self.document['run']['completed_tasks'] == 1)

    def checkpoint(self, force=False):
        """Save after ten completions or five seconds, using atomic replacement."""
        if not self.dirty or not (force or self.since_checkpoint >= 10 or time.monotonic() - self.last_checkpoint >= 5):
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent, delete=False) as fp:
                temporary = fp.name
                json.dump(self.document, fp, indent=2, ensure_ascii=False)
                fp.write('\n')
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        self.last_checkpoint = time.monotonic()
        self.since_checkpoint = 0
        self.dirty = False

    def __exit__(self, kind, error, tb):
        self.document['run'].update(status='interrupted' if kind in (KeyboardInterrupt, asyncio.CancelledError) else 'failed' if kind else 'completed', finished_at=utc_now())
        if error is not None:
            self.document['run']['error'] = str(error)
        self.dirty = True
        self.checkpoint(force=True)


@contextmanager
def capture_console_log(path):
    """Capture stdout/stderr to a file and mirror it to the console, including children."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Existing shell redirection to this same file must not create a second writer.
    if path.exists() and (os.stat(path).st_dev, os.stat(path).st_ino) == (os.fstat(1).st_dev, os.fstat(1).st_ino):
        saved_err = os.dup(2)
        os.dup2(1, 2)
        try:
            yield
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stderr.flush()
            os.dup2(saved_err, 2)
            os.close(saved_err)
        return
    original_out, original_err = os.dup(1), os.dup(2)
    stopped = threading.Event()
    final_size = 0
    mirror_errors = []

    def mirror_to_console():
        try:
            with path.open('rb') as reader, os.fdopen(os.dup(original_out), 'wb') as console:
                while not stopped.is_set():
                    chunk = reader.read(65536)
                    if chunk:
                        console.write(chunk)
                        console.flush()
                    else:
                        stopped.wait(0.05)
                # Drain exactly the bytes present at shutdown, regardless of child writers.
                remaining = final_size - reader.tell()
                while remaining > 0:
                    chunk = reader.read(min(remaining, 65536))
                    if not chunk:
                        break
                    console.write(chunk)
                    remaining -= len(chunk)
        except BrokenPipeError:
            # A console consumer such as head may exit before the benchmark does.
            pass
        except Exception as exc:
            mirror_errors.append(exc)

    try:
        with path.open('wb') as log:
            mirror = threading.Thread(target=mirror_to_console, name='console-log-mirror')
            mirror.start()
            try:
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                try:
                    yield
                except BaseException:
                    traceback.print_exc()
                    raise
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    os.dup2(original_out, 1)
                    os.dup2(original_err, 2)
                    final_size = os.fstat(log.fileno()).st_size
                    stopped.set()
                    mirror.join()
                if mirror_errors:
                    if sys.exc_info()[0] is None:
                        raise RuntimeError("Console log mirroring failed") from mirror_errors[0]
                    print(f"Console log mirroring failed: {mirror_errors[0]}", file=sys.stderr)
    finally:
        os.close(original_out)
        os.close(original_err)
