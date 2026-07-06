#!/usr/bin/env bash
# build_power_tiles.sh — OSM power features → PMTiles (DATACORE MAXIMUS
# Phase 2, GRID BUILD ORDER item 1). PROVEN END-TO-END 2026-07-06 in the
# session container: texas 709MB PBF → 5.2MB filtered (31s) → 111MB
# GeoJSONSeq → 16.2MB PMTiles (25s, tippecanoe chose maxzoom 12).
#
# Prereqs (one apt call): apt-get install -y osmium-tool tippecanoe
# Usage: ./scripts/build_power_tiles.sh [region-path] [out-name]
#   default region: north-america/us/texas
#   e.g. US full:   ./scripts/build_power_tiles.sh north-america/us power_us
#        (12GB download, ≲1hr total, ~15GB peak disk — delete PBF after)
#
# ODbL: output tiles are a PRODUCED WORK — attribution
# "© OpenStreetMap contributors, ODbL" required on the layer; do NOT
# offer the .pmtiles file itself for download (share-alike boundary).
# VOLTAGE HONESTY: the voltage tag survives into tile features; render
# untagged-voltage lines as a distinct "voltage unknown" class — never
# silently drop them (OSM voltage coverage is incomplete).
set -euo pipefail

REGION="${1:-north-america/us/texas}"
OUT="${2:-power_tx}"
WORK="${TMPDIR:-/tmp}/power_tiles_build"
mkdir -p "$WORK"
cd "$WORK"

PBF="$(basename "$REGION")-latest.osm.pbf"
if [ ! -f "$PBF" ]; then
  echo "downloading https://download.geofabrik.de/${REGION}-latest.osm.pbf ..."
  curl -sSL -o "$PBF" "https://download.geofabrik.de/${REGION}-latest.osm.pbf"
fi

echo "filtering power features (streaming, low RAM) ..."
osmium tags-filter "$PBF" \
  w/power=line,minor_line,cable \
  nwr/power=substation,plant \
  -o power_filtered.osm.pbf --overwrite

echo "exporting GeoJSONSeq ..."
osmium export power_filtered.osm.pbf -o power.geojsonseq -f geojsonseq --overwrite

echo "tiling (tippecanoe writes .pmtiles natively) ..."
tippecanoe -o "${OUT}.pmtiles" -l power -zg \
  --drop-densest-as-needed --extend-zooms-if-still-dropping \
  --force power.geojsonseq

ls -la "${OUT}.pmtiles"
echo "DONE: $WORK/${OUT}.pmtiles"
echo "Serving decision (see wishlist DATACORE MAXIMUS block):"
echo "  TX pilot (16MB): commit under client/public/tiles/ -> bundled to"
echo "  dist/public, served with range requests by express static."
echo "  US scale (est. 60-150MB): DO NOT commit — boot-fetch from a"
echo "  GitHub Release asset into the volume, serve from there."
