# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/segmentation/segment_tile.py

"""
Wrapper for the C++ TreeSeparation segmentation binary.

Reads:
    data/<case>/tiles/<tile_id>/vegetation.xyz
Writes:
    data/<case>/tiles/<tile_id>/segmentation.xyz
    data/<case>/tiles/<tile_id>/tree_hulls.geojson

Returns:
    {"tile_id": str, "status": "ok"|"skipped"|"failed", "outputs": dict}
"""

from __future__ import annotations
import logging
import subprocess
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPoint

from src.config import get_config


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# Parameters passed to segmentation binary
SEG_PARAMS = {
    "radius": 2.5,
    "vres": 1.5,
    "min_pts": 3,
}
cfg = get_config()


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def segment_tile(tile_dir: Path, overwrite: bool = False) -> dict:
    """
    Run TreeSeparation C++ segmentation on one tile directory.

    Reads:
        vegetation.xyz
    Writes:
        segmentation.xyz, tree_hulls.geojson
    """
    tile_id = tile_dir.name
    input_xyz = tile_dir / "vegetation.xyz"
    output_xyz = tile_dir / "segmentation.xyz"
    hulls_geojson = tile_dir / "tree_hulls.geojson"

    exe = Path(__file__).parent / "TreeSeparation" / "build" / "segmentation"

    # Pre-checks
    if not input_xyz.exists():
        logging.warning(f"[{tile_id}] Missing input vegetation.xyz — skipping segmentation.")
        return {"tile_id": tile_id, "status": "missing_input", "outputs": {}}

    if not exe.exists():
        logging.error(f"[{tile_id}] Missing C++ segmentation binary: {exe}")
        return {"tile_id": tile_id, "status": "missing_binary", "outputs": {}}

    if output_xyz.exists() and hulls_geojson.exists() and not overwrite:
        logging.info(f"[{tile_id}] Segmentation already exists — skipping (use --overwrite to redo).")
        return {"tile_id": tile_id, "status": "skipped", "outputs": {"seg_xyz": output_xyz, "hulls": hulls_geojson}}

    # ------------------------------------------------------------------
    # Run segmentation binary
    # ------------------------------------------------------------------
    cmd = [
        str(exe),
        str(input_xyz),
        str(output_xyz),
        str(SEG_PARAMS["radius"]),
        str(SEG_PARAMS["vres"]),
        str(SEG_PARAMS["min_pts"]),
    ]

    logging.info(f"[{tile_id}] Running segmentation binary...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip()
        logging.warning(f"[{tile_id}] Segmentation failed: {stderr}")
        return {"tile_id": tile_id, "status": "failed", "outputs": {}}

    # ------------------------------------------------------------------
    # Post-processing: build convex hulls per tree
    # ------------------------------------------------------------------
    try:
        seg_df = pd.read_csv(output_xyz, sep=r"\s+", header=None, names=["tid", "x", "y", "z"])
        seg_gdf = gpd.GeoDataFrame(seg_df, geometry=gpd.points_from_xy(seg_df.x, seg_df.y), crs=cfg["crs"])

        hulls = []
        for tid, group in seg_gdf.groupby("tid"):
            if len(group) >= 3:
                hull_geom = MultiPoint(group.geometry.values).convex_hull
                hulls.append({"tid": tid, "geometry": hull_geom})
            else:
                logging.debug(f"[{tile_id}] Tree ID {tid} has <3 points — skipped.")

        if hulls:
            hulls_gdf = gpd.GeoDataFrame(hulls, crs=cfg["crs"])
            hulls_gdf.to_file(hulls_geojson, driver="GeoJSON")
        else:
            logging.warning(f"[{tile_id}] No valid hulls produced.")

        logging.info(f"[{tile_id}] Segmentation complete.")
        return {
            "tile_id": tile_id,
            "status": "ok",
            "outputs": {"seg_xyz": output_xyz, "hulls": hulls_geojson},
        }

    except Exception as e:
        logging.warning(f"[{tile_id}] Post-processing failed: {e}")
        return {"tile_id": tile_id, "status": "failed", "outputs": {}}
