# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/tiles_clipper_robust.sh

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Clip a single LAZ file using PDAL crop filter.
#
# Usage:
#   bash src/get_data/tiles_clipper_robust.sh <input_laz> <aoi_geojson> <output_laz>
#
# Example:
#   bash src/get_data/tiles_clipper_robust.sh \
#       data/wippolder/tiles/37EN2_11/raw.laz \
#       cases/wippolder/city_bbox_buffered.geojson \
#       data/wippolder/tiles/37EN2_11/clipped.laz
#
# The AOI is automatically converted to WKT.
# Output is written to the specified clipped LAZ path.
# ---------------------------------------------------------------------------

set -euo pipefail

INPUT_LAZ="${1:-}"
AOI_GEOJSON="${2:-}"
OUTPUT_LAZ="${3:-}"

if [[ -z "$INPUT_LAZ" || -z "$AOI_GEOJSON" || -z "$OUTPUT_LAZ" ]]; then
    echo "Usage: $0 <input_laz> <aoi_geojson> <output_laz>"
    exit 1
fi

if [[ ! -f "$INPUT_LAZ" ]]; then
    echo "ERROR: Input LAZ file not found: $INPUT_LAZ"
    exit 2
fi

if [[ ! -f "$AOI_GEOJSON" ]]; then
    echo "ERROR: AOI GeoJSON not found: $AOI_GEOJSON"
    exit 3
fi

TILE_ID="$(basename "$(dirname "$INPUT_LAZ")")"
TMP_DIR="/tmp/tmp_pdal_${TILE_ID}_$$"
mkdir -p "$TMP_DIR"

# Convert polygon to WKT
WKT_FILE="$TMP_DIR/aoi.wkt"
ogrinfo -geom=YES -al "$AOI_GEOJSON" | grep POLYGON | head -n 1 | sed 's/^[ \t]*//' > "$WKT_FILE"

if [[ ! -s "$WKT_FILE" ]]; then
    echo "ERROR: Failed to extract WKT from $AOI_GEOJSON"
    rm -rf "$TMP_DIR"
    exit 4
fi

# Build PDAL pipeline
PIPELINE_FILE="$TMP_DIR/${TILE_ID}_clip.json"

cat > "$PIPELINE_FILE" <<EOF
{
  "pipeline": [
    "$INPUT_LAZ",
    {
      "type": "filters.crop",
      "polygon": "$(cat "$WKT_FILE")"
    },
    {
      "type": "writers.las",
      "filename": "$OUTPUT_LAZ",
      "compression": true,
      "minor_version": 4,
      "dataformat_id": 8
    }
  ]
}
EOF

# Run PDAL clipping
echo "[$TILE_ID] Clipping..."
pdal pipeline "$PIPELINE_FILE" > "$TMP_DIR/pdal.log" 2>&1 || {
    echo "[$TILE_ID] ERROR: PDAL clipping failed — see $TMP_DIR/pdal.log"
    rm -rf "$TMP_DIR"
    exit 5
}

# Cleanup
rm -rf "$TMP_DIR"
echo "[$TILE_ID] Done: clipped to $OUTPUT_LAZ"
