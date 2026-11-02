# Self-hosted GraphHopper routing, built from a PBF file via config.yml.
#
# Usage:
#   make build    # Build the graph-cache (one-time)
#   make up       # Start the GraphHopper server
#   make down     # Stop the GraphHopper server

GH_VERSION ?= 11.1
BUILD_DIR := build
GH_JAR := $(BUILD_DIR)/graphhopper-web-$(GH_VERSION).jar
GH_JAR_URL := https://repo1.maven.org/maven2/com/graphhopper/graphhopper-web/$(GH_VERSION)/graphhopper-web-$(GH_VERSION).jar
GH_PORT ?= 8989
GH_ADMIN_PORT ?= 8990
PIDFILE := $(BUILD_DIR)/.graphhopper.pid
LOGFILE := $(BUILD_DIR)/graphhopper.log

.PHONY: build up down clean

$(GH_JAR):
	@mkdir -p $(BUILD_DIR)
	curl -L -o $(GH_JAR) $(GH_JAR_URL)

# One-time: builds build/graph-cache/ from the pbf for all profiles in config.yml. Skips if present.
build: $(GH_JAR)
	@if [ -d $(BUILD_DIR)/graph-cache ]; then \
		echo "[graphhopper:build] $(BUILD_DIR)/graph-cache/ already exists, skipping"; \
	else \
		java -jar $(GH_JAR) import config.yml; \
	fi

# nohup-start graphhopper serving on $(GH_PORT), pidfile-based
up:
	@if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "[graphhopper:up] already running (pid $$(cat $(PIDFILE)))"; \
	else \
		nohup java -jar $(GH_JAR) server config.yml > $(LOGFILE) 2>&1 & echo $$! > $(PIDFILE); \
	fi
	@echo "[graphhopper:up] waiting for http://127.0.0.1:$(GH_ADMIN_PORT)/healthcheck ..."
	@for i in $$(seq 1 60); do \
		if curl -sf "http://127.0.0.1:$(GH_ADMIN_PORT)/healthcheck" >/dev/null 2>&1; then \
			echo "[graphhopper:up] ready"; exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "[graphhopper:up] FAILED to come up, check $(LOGFILE)"; exit 1

down:
	@if [ -f $(PIDFILE) ]; then \
		kill $$(cat $(PIDFILE)) 2>/dev/null && echo "[graphhopper:down] stopped" || echo "[graphhopper:down] not running"; \
		rm -f $(PIDFILE); \
	else \
		echo "[graphhopper:down] no pidfile, not running"; \
	fi

clean: down
	rm -rf $(BUILD_DIR)
