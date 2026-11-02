# Sets up everything this MCP server (server.ts) needs: the Postgres/PostGIS database
# (planet_osm_* via osm2pgsql) plus the Photon/GraphHopper services it calls over HTTP.
# Run once per machine/dataset, from this directory:
#
#   make
#
# or the individual steps, in order:
#
#   make init_db load_db services_build services_up
#
# Thereafter, start/stop just the services (DB stays up on its own):
#
#   make services_up / make services_down
#
# Full wipe (drops the DB, removes both services' build output):
#
#   make clean

PBF_FILE ?= data/west_lafayette.pbf

DB_NAME ?= osm
DB_USER ?= $(USER)
DB_HOST ?= localhost
DB_PORT ?= 5432
PGCONN := --host $(DB_HOST) --port $(DB_PORT) --username $(DB_USER)

.PHONY: all clean init_db load_db \
	services_build services_up services_down services_clean \
	photon_build photon_up photon_down photon_clean \
	graphhopper_build graphhopper_up graphhopper_down graphhopper_clean

# Run make to build the PostGIS database and both services, then start the services.
all: init_db load_db services_build services_up

# Full wipe: drops the database and removes both services' build output.
clean: services_clean
	dropdb $(PGCONN) --if-exists $(DB_NAME)

# (Re)creates PostGIS database $(DB_NAME), owned by $(DB_USER), with extensions below.
init_db:
	dropdb $(PGCONN) --if-exists $(DB_NAME)
	createdb $(PGCONN) $(DB_NAME)
	psql $(PGCONN) -X -v ON_ERROR_STOP=1 $(DB_NAME) -c "CREATE EXTENSION IF NOT EXISTS postgis;"
	psql $(PGCONN) -X -v ON_ERROR_STOP=1 $(DB_NAME) -c "CREATE EXTENSION IF NOT EXISTS hstore;"

# Load OSM PBF data into planet_osm_* tables {points, lines, polygons} + metadata tables.
load_db:
	osm2pgsql --create --drop --latlong --hstore --multi-geometry \
		$(PGCONN) --database $(DB_NAME) $(PBF_FILE)

# Thin forwarders to services/{photon,graphhopper}/Makefile
services_build: photon_build graphhopper_build
services_up: photon_up graphhopper_up
services_down: photon_down graphhopper_down
services_clean: photon_clean graphhopper_clean

photon_build:
	$(MAKE) -C services/photon build
photon_up:
	$(MAKE) -C services/photon up
photon_down:
	$(MAKE) -C services/photon down
photon_clean:
	$(MAKE) -C services/photon clean

graphhopper_build:
	$(MAKE) -C services/graphhopper build
graphhopper_up:
	$(MAKE) -C services/graphhopper up
graphhopper_down:
	$(MAKE) -C services/graphhopper down
graphhopper_clean:
	$(MAKE) -C services/graphhopper clean
