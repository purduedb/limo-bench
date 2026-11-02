# The Python CLIs own runner defaults; ARGS passes additional flags directly.
.DEFAULT_GOAL := all
PYTHON ?= python
export PYTHONUNBUFFERED := 1

ifneq (,$(wildcard .env))
include .env
endif

RUN_ID ?= latest
OUTPUT_ROOT ?= outputs
RUN_DIR = $(OUTPUT_ROOT)/$(RUN_ID)
TASK_FILE ?= data/west_lafayette/v1/benchmark_tasks.json
PROVIDER ?= openrouter
MODEL ?= openai/gpt-6-luna
REASONING ?= low
MAX_RETRIES ?= 8
LIMIT ?=
OPERATION ?=
ARGS ?=
BASELINE_ARGS ?= --inter-task-delay 0.25
MCP_RUN_ARGS ?= --inter-task-delay 0.35

COMMON_ARGS = --default-ip-location --tasks "$(TASK_FILE)" \
    --provider "$(PROVIDER)" --model "$(MODEL)" --max-retries $(MAX_RETRIES) \
    $(if $(REASONING),--reasoning "$(REASONING)") \
    $(if $(LIMIT),--limit $(LIMIT)) $(if $(OPERATION),--operation "$(OPERATION)") --verbose

# MCP launch presets. Override MCP_COMMAND, MCP_ARGS, and MCP_ENV for another server.
MCP_VENDOR ?= purduedb/limo

# MCP server launch/run command per vendor
# Spawned as: subprocess with env=<env-dict>, argv=[<command>, *<args>] in src/mcp_client.py
ifeq ($(MCP_VENDOR),purduedb/limo)
  # Build step: npm ci. Always run (warm ~1s, cold ~3s) so edits are never stale.
  MCP_BUILD ?= cd vendor/purduedb__limo && npm ci

  # Needs its DB set up first - cd vendor/purduedb__limo && make
  # Also needs Photon/GraphHopper up (self-hosted geocoding/routing) -
  #   cd vendor/purduedb__limo/services/photon && make build up
  #   cd vendor/purduedb__limo/services/graphhopper && make build up

  # Launch step
  DB_NAME ?= osm
  DB_USER ?= $(USER)
  DB_HOST ?= localhost
  DB_PORT ?= 5432
  MCP_ENV ?= DB_NAME=$(DB_NAME) DB_USER=$(DB_USER) DB_HOST=$(DB_HOST) DB_PORT=$(DB_PORT)
  MCP_COMMAND ?= vendor/purduedb__limo/node_modules/.bin/tsx
  MCP_ARGS ?= vendor/purduedb__limo/server.ts
endif
export DB_NAME DB_USER DB_HOST DB_PORT

.PHONY: all archive baseline mcp mcp_serve eval test test_baseline test_mcp test_tools viewer

# Recursive calls keep archive and the two conditions ordered, including with make -j.
all:
	$(MAKE) archive
	$(MAKE) baseline
	$(MAKE) mcp

archive:
	@if [ "$(RUN_ID)" = latest ] && [ -d "$(RUN_DIR)" ]; then \
		destination=$$(mktemp -d "$(OUTPUT_ROOT)/$$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX"); \
		rmdir "$$destination" && mv "$(RUN_DIR)" "$$destination"; \
	fi

baseline:
	$(PYTHON) src/run_baseline.py \
		$(COMMON_ARGS) \
		--output "$(RUN_DIR)/baseline/run.json" \
		$(BASELINE_ARGS) \
		$(ARGS)

# Keep build logs off stdout: stdio MCP reserves it for protocol messages.
mcp_serve:
	@( $(if $(strip $(MCP_BUILD)),$(MCP_BUILD),:) ) >&2
	@exec env $(MCP_ENV) $(MCP_COMMAND) $(MCP_ARGS)

mcp:
	$(PYTHON) src/run_mcp.py \
		$(COMMON_ARGS) \
		--output "$(RUN_DIR)/mcp/run.json" \
		--mcp-vendor "$(MCP_VENDOR)" \
		--mcp-command "$(MAKE)" \
		--mcp-arg=--silent \
		--mcp-arg=--no-print-directory \
		--mcp-arg=mcp_serve \
		--mcp-arg="MCP_VENDOR=$(MCP_VENDOR)" \
		$(MCP_RUN_ARGS) \
		$(ARGS)

eval:
	$(PYTHON) src/evaluate.py \
		"$(RUN_DIR)/baseline/run.json" \
		"$(RUN_DIR)/mcp/run.json" \
		--csv "$(RUN_DIR)/eval.csv" \
		$(ARGS)

# Offline checks never call a paid model API.
test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

# Live smoke tests use the same CLI as complete runs.
test_baseline:
	$(MAKE) baseline RUN_ID=test LIMIT=$(or $(LIMIT),1)

test_mcp:
	$(MAKE) mcp RUN_ID=test LIMIT=$(or $(LIMIT),1)

test_tools:
	@mkdir -p "$(OUTPUT_ROOT)/test_tools"
	$(PYTHON) -c "import json,pathlib; data=json.loads(pathlib.Path('$(TASK_FILE)').read_text()); tasks=data.get('tasks',[]) if isinstance(data,dict) else data; seen={}; [seen.setdefault(t['operation'],t) for t in sorted(tasks,key=lambda t:t['task_id'])]; pathlib.Path('$(OUTPUT_ROOT)/test_tools/tasks.json').write_text(json.dumps(list(seen.values()),indent=2))"
	$(MAKE) mcp RUN_ID=test_tools TASK_FILE="$(OUTPUT_ROOT)/test_tools/tasks.json"

VIEWER_DIR ?= data/west_lafayette/v1
viewer:
	$(PYTHON) -c "import functools,http.server as h,webbrowser; s=h.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(h.SimpleHTTPRequestHandler,directory='$(VIEWER_DIR)')); url=f'http://localhost:{s.server_port}/viewer.html'; print(url,flush=True); webbrowser.open(url); s.serve_forever()"
