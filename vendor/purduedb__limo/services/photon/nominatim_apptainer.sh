#!/usr/bin/env bash
# One-time Nominatim import + Photon import with Apptainer (no Docker, no root).
# Called by `make build` when Docker is unavailable. Inputs come from the Makefile:
#   PBF_FILE PHOTON_JAR PHOTON_DATA BUILD_DIR NOMINATIM_IMAGE NOMINATIM_PG_PORT NOMINATIM_PASSWORD
# The image's own start script needs root (useradd, sudo, service), so this runs the
# image's Postgres and `nominatim` directly as the current user.
set -euo pipefail

WORK="$(cd "$BUILD_DIR" && pwd)/nominatim_apptainer"
SIF="$BUILD_DIR/nominatim.sif"
PORT="$NOMINATIM_PG_PORT"
mkdir -p "$WORK/project"

[ -f "$SIF" ] || apptainer pull "$SIF" "docker://$NOMINATIM_IMAGE"
cp "$PBF_FILE" "$WORK/project/data.osm.pbf"
run() { apptainer exec --bind "$WORK" "$SIF" "$@"; }

echo "[photon:build] nominatim import via apptainer (this takes a while)..."
run bash -euc '
	export PATH="$(ls -d /usr/lib/postgresql/*/bin | tail -1):$PATH"
	W=$1; PORT=$2
	rm -f "$W/pg/postmaster.pid"
	[ -f "$W/pg/PG_VERSION" ] || initdb -D "$W/pg" -E UTF8 --locale=C.UTF-8 -U nominatim >/dev/null
	postgres -D "$W/pg" -p "$PORT" -k /tmp -c listen_addresses=127.0.0.1 \
		-c fsync=off -c synchronous_commit=off -c full_page_writes=off &
	PG=$!
	until pg_isready -q -U nominatim -h 127.0.0.1 -p "$PORT"; do sleep 1; done
	createuser -h 127.0.0.1 -p "$PORT" -U nominatim www-data 2>/dev/null || true
	dropdb -h 127.0.0.1 -p "$PORT" -U nominatim --if-exists nominatim
	cd "$W/project"
	NOMINATIM_DATABASE_DSN="pgsql:dbname=nominatim;host=127.0.0.1;port=$PORT;user=nominatim" \
		nominatim import --osm-file data.osm.pbf --threads "$(nproc)"
	kill "$PG"; wait "$PG" || true
' _ "$WORK" "$PORT"

echo "[photon:build] nominatim ready, running photon import..."
run bash -euc '
	export PATH="$(ls -d /usr/lib/postgresql/*/bin | tail -1):$PATH"
	exec postgres -D "$1/pg" -p "$2" -k /tmp -c listen_addresses=127.0.0.1
' _ "$WORK" "$PORT" &
PG=$!
stop_pg() { run bash -c "PATH=\$(ls -d /usr/lib/postgresql/*/bin | tail -1):\$PATH; pg_ctl -D '$WORK/pg' -m fast -w stop" >/dev/null 2>&1 || true; wait "$PG" 2>/dev/null || true; }
trap stop_pg EXIT
until (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do kill -0 "$PG"; sleep 1; done

# UTC: the host's zone can be a legacy alias (e.g. America/Indianapolis) that the image's Postgres rejects.
java -Duser.timezone=UTC -jar "$PHOTON_JAR" -data-dir "$PHOTON_DATA" \
	import -host 127.0.0.1 -port "$PORT" -database nominatim -user nominatim -password "$NOMINATIM_PASSWORD"

stop_pg
trap - EXIT
rm -rf "$WORK" || true
