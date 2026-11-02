# Self-hosted Photon geocoder, built from a PBF file.
#
# Photon only imports from a Nominatim DB, so the build step runs a disposable
# Nominatim container to import the pbf, then tears it down. Uses Docker when
# available, else Apptainer (rootless; see nominatim_apptainer.sh). Force one with
# NOMINATIM_RUNTIME=docker|apptainer.
#
# Usage:
#   make build    # Build the photon index (one-time)
#   make up       # Start the Photon server
#   make down     # Stop the Photon server

PBF_FILE ?= $(abspath ../../data/west_lafayette.pbf)
BUILD_DIR := build
PHOTON_VERSION ?= 1.3.0
PHOTON_JAR := $(BUILD_DIR)/photon-$(PHOTON_VERSION).jar
PHOTON_JAR_URL := https://github.com/komoot/photon/releases/download/$(PHOTON_VERSION)/photon-$(PHOTON_VERSION).jar
NOMINATIM_IMAGE ?= mediagis/nominatim:5.3
NOMINATIM_RUNTIME ?= $(if $(shell docker info >/dev/null 2>&1 && echo y),docker,apptainer)
NOMINATIM_CONTAINER ?= limo-nominatim
NOMINATIM_PASSWORD ?= limo_nominatim_pw
# Non-standard host port - 5432 is taken by this project's own PostGIS instance.
NOMINATIM_PG_PORT ?= 15432
PHOTON_PORT ?= 2322
PHOTON_DATA := $(abspath $(BUILD_DIR)/photon_data)
NOMINATIM_PBF_DIR := $(BUILD_DIR)/nominatim_pbf
PIDFILE := $(BUILD_DIR)/.photon.pid
LOGFILE := $(BUILD_DIR)/photon.log

.PHONY: build up down clean

$(PHOTON_JAR):
	@mkdir -p $(BUILD_DIR)
	curl -L -o $(PHOTON_JAR) $(PHOTON_JAR_URL)

# One-time: nominatim imports the pbf into its own postgres, then photon imports from
# that nominatim db into build/photon_data. Skips if photon_data already exists.
build: $(PHOTON_JAR)
	@if [ -d "$(PHOTON_DATA)" ]; then \
		echo "[photon:build] $(PHOTON_DATA) already exists, skipping"; \
	elif [ "$(NOMINATIM_RUNTIME)" = apptainer ]; then \
		mkdir -p "$(PHOTON_DATA)"; \
		BUILD_DIR="$(BUILD_DIR)" PBF_FILE="$(PBF_FILE)" PHOTON_JAR="$(PHOTON_JAR)" PHOTON_DATA="$(PHOTON_DATA)" \
			NOMINATIM_IMAGE="$(NOMINATIM_IMAGE)" NOMINATIM_PG_PORT="$(NOMINATIM_PG_PORT)" NOMINATIM_PASSWORD="$(NOMINATIM_PASSWORD)" \
			./nominatim_apptainer.sh || { rm -rf "$(PHOTON_DATA)"; exit 1; }; \
		echo "[photon:build] done, index in $(PHOTON_DATA)"; \
	else \
		echo "[photon:build] starting nominatim, importing $(PBF_FILE) (this takes a while)..."; \
		docker rm -f $(NOMINATIM_CONTAINER) >/dev/null 2>&1 || true; \
		mkdir -p $(NOMINATIM_PBF_DIR); \
		cp "$(PBF_FILE)" $(NOMINATIM_PBF_DIR)/west_lafayette.pbf; \
		docker run -d --shm-size=1g \
			-e PBF_PATH=/nominatim/data/west_lafayette.pbf \
			-e NOMINATIM_PASSWORD=$(NOMINATIM_PASSWORD) \
			-v "$(abspath $(NOMINATIM_PBF_DIR))":/nominatim/data \
			-p 127.0.0.1:$(NOMINATIM_PG_PORT):5432 \
			-p 127.0.0.1:8080:8080 \
			--name $(NOMINATIM_CONTAINER) \
			$(NOMINATIM_IMAGE); \
		echo "[photon:build] waiting for nominatim import to finish..."; \
		until curl -sf http://127.0.0.1:8080/status.php >/dev/null 2>&1; do \
			if [ "$$(docker inspect -f '{{.State.Running}}' $(NOMINATIM_CONTAINER) 2>/dev/null)" != "true" ]; then \
				echo "[photon:build] nominatim container exited, check: docker logs $(NOMINATIM_CONTAINER)"; exit 1; \
			fi; \
			sleep 5; \
		done; \
		echo "[photon:build] nominatim ready, running photon import..."; \
		mkdir -p "$(PHOTON_DATA)"; \
		java -Duser.timezone=UTC -jar $(PHOTON_JAR) -data-dir "$(PHOTON_DATA)" \
			import -host 127.0.0.1 -port $(NOMINATIM_PG_PORT) -database nominatim -user nominatim -password $(NOMINATIM_PASSWORD); \
		docker stop $(NOMINATIM_CONTAINER) >/dev/null; \
		docker rm $(NOMINATIM_CONTAINER) >/dev/null; \
		rm -rf $(NOMINATIM_PBF_DIR); \
		echo "[photon:build] done, index in $(PHOTON_DATA)"; \
	fi

# nohup-start photon serving on $(PHOTON_PORT), pidfile-based
up:
	@if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "[photon:up] already running (pid $$(cat $(PIDFILE)))"; \
	else \
		nohup java -jar $(PHOTON_JAR) -data-dir "$(PHOTON_DATA)" \
			-listen-port $(PHOTON_PORT) serve > $(LOGFILE) 2>&1 & echo $$! > $(PIDFILE); \
	fi
	@echo "[photon:up] waiting for http://127.0.0.1:$(PHOTON_PORT) ..."
	@for i in $$(seq 1 60); do \
		if curl -sf "http://127.0.0.1:$(PHOTON_PORT)/api?q=test&limit=1" >/dev/null 2>&1; then \
			echo "[photon:up] ready"; exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "[photon:up] FAILED to come up, check $(LOGFILE)"; exit 1

down:
	@if [ -f $(PIDFILE) ]; then \
		kill $$(cat $(PIDFILE)) 2>/dev/null && echo "[photon:down] stopped" || echo "[photon:down] not running"; \
		rm -f $(PIDFILE); \
	else \
		echo "[photon:down] no pidfile, not running"; \
	fi

clean: down
	rm -rf $(BUILD_DIR)
