"""
src/segmentation/generalize_forest_ids.py

Assigns global tree IDs (GTIDs) to segmented trees across all tiles of a case,
removes trees outside the AOI, and enriches vegetation.laz with GTIDs.

Reads:
    cases/<case>/city_bbox.geojson
    data/<case>/tiles/<tile_id>/tree_hulls.geojson
    data/<case>/tiles/<tile_id>/segmentation.xyz
    data/<case>/tiles/<tile_id>/vegetation.laz

Writes:
    data/<case>/forest_hulls.geojson
    data/<case>/gtid_map.csv
    data/<case>/tiles/<tile_id>/forest.laz
"""

from __future__ import annotations
import hashlib
import logging
from pathlib import Path
import geopandas as gpd
import pandas as pd
import laspy
from shapely.geometry import Point
from src.config import get_config
import numpy as np

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def case_prefix_hash(case: str) -> int:
    """Deterministic 3-digit numeric hash of case name."""
    return int(hashlib.sha1(case.encode()).hexdigest(), 16) % 1000


def compute_gtid(case_prefix: int, counter: int) -> int:
    """Combine 3-digit prefix with 7-digit sequential counter."""
    return int(f"{case_prefix:03d}{counter:07d}")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def generalize_forest_ids(case: str, overwrite: bool = False) -> dict:
    """
    Create global tree IDs (GTIDs) and write forest.laz per tile.

    Returns:
        dict with summary info and output paths.
    """
    cfg = get_config()
    data_root = cfg["data_root"]
    case_dir = data_root / case
    tiles_dir = case_dir / "tiles"

    # ------------------------------------------------------------------
    # Load AOI polygon
    # ------------------------------------------------------------------
    aoi_path = Path("cases") / case / "city_bbox.geojson"
    if not aoi_path.exists():
        logging.error(f"AOI not found: {aoi_path}")
        return {"case": case, "status": "missing_aoi"}

    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        logging.error(f"AOI file is empty: {aoi_path}")
        return {"case": case, "status": "empty_aoi"}

    aoi_geom = aoi.to_crs(cfg["crs"]).geometry.unary_union
    case_prefix = case_prefix_hash(case)
    logging.info(f"Case hash prefix: {case_prefix:03d}")

    # ------------------------------------------------------------------
    # Collect all tree hulls within AOI
    # ------------------------------------------------------------------
    hulls_all = []
    for tile_dir in sorted(tiles_dir.iterdir()):
        logging.debug(f"Entering tile: {tile_dir}")

        hull_path = tile_dir / "tree_hulls.geojson"
        if not hull_path.exists():
            continue
        try:
            gdf = gpd.read_file(hull_path).to_crs(cfg["crs"])
            gdf["tile_id"] = tile_dir.name
            gdf["centroid"] = gdf.geometry.centroid
            gdf = gdf[gdf["centroid"].within(aoi_geom)]
            if not gdf.empty:
                hulls_all.append(gdf)
        except Exception as e:
            logging.warning(f"[{tile_dir.name}] Failed reading hulls: {e}")

    if not hulls_all:
        logging.warning(f"No valid tree hulls found inside AOI for case {case}")
        return {"case": case, "status": "no_hulls"}

    hulls = pd.concat(hulls_all, ignore_index=True)
    hulls = hulls.drop(columns="centroid")
    hulls = gpd.GeoDataFrame(hulls, crs=cfg["crs"])

    # ------------------------------------------------------------------
    # Assign GTIDs sequentially
    # ------------------------------------------------------------------
    hulls["gtid"] = [compute_gtid(case_prefix, i + 1) for i in range(len(hulls))]
    n_trees = len(hulls)
    logging.info(f"Assigned GTIDs for {n_trees} trees across {len(hulls_all)} tiles.")

    # ------------------------------------------------------------------
    # Write forest-level outputs
    # ------------------------------------------------------------------
    out_forest_hulls = case_dir / "forest_hulls.geojson"
    out_gtid_map = case_dir / "gtid_map.csv"

    hulls[["tile_id", "tid", "gtid", "geometry"]].to_file(out_forest_hulls, driver="GeoJSON")
    hulls[["tile_id", "tid", "gtid"]].to_csv(out_gtid_map, index=False)
    logging.info(f"Wrote forest hulls: {out_forest_hulls}")
    logging.info(f"Wrote GTID map: {out_gtid_map}")

    # ------------------------------------------------------------------
    # Enrich vegetation.laz per tile with GTID
    # ------------------------------------------------------------------
    gtid_map = pd.read_csv(out_gtid_map)
    for tile_dir in sorted(tiles_dir.iterdir()):
        veg_path = tile_dir / "vegetation.laz"
        seg_path = tile_dir / "segmentation.xyz"
        out_forest = tile_dir / "forest.laz"

        if not veg_path.exists() or not seg_path.exists():
            logging.debug(f"[{tile_dir.name}] Missing vegetation or segmentation file — skipped.")
            continue
        if out_forest.exists() and not overwrite:
            logging.debug(f"[{tile_dir.name}] Forest already exists — skipped.")
            continue

        # Load segmentation table
        try:
            logging.debug(f"Reading segmentation file: {seg_path}")
            seg_df = pd.read_csv(seg_path, sep=r"\s+", header=None, names=["tid", "x", "y", "z"])
            logging.debug(f"Segmentation rows: {len(seg_df)}")
        except Exception as e:
            logging.warning(f"[{tile_dir.name}] Failed reading segmentation.xyz: {e}")
            continue

        # Join with GTID map
        tile_map = gtid_map[gtid_map["tile_id"] == tile_dir.name]
        if tile_map.empty:
            logging.debug(f"[{tile_dir.name}] No GTIDs found for this tile — skipped.")
            continue

        seg_df = seg_df.merge(tile_map, on="tid", how="inner")
        if seg_df.empty:
            logging.debug(f"[{tile_dir.name}] No matching GTIDs after merge — skipped.")
            continue


        # Read vegetation.laz
        try:
            logging.debug(f"Reading vegetation LAS: {veg_path}")
            with laspy.open(veg_path) as src:
                las = src.read()
            if not all(len(np.asarray(arr)) > 0 for arr in [las.x, las.y, las.z]):
                logging.debug(f"[{tile_dir.name}] LAS has no coordinate data — skipped.")
                continue

        except Exception as e:
            logging.warning(f"[{tile_dir.name}] Failed reading vegetation.laz: {e}")
            continue

        # Build spatial index to match by coordinates (approximate)
        logging.debug(f"[{tile_dir.name}] Building vegetation DataFrame for merging.")
        veg_df = pd.DataFrame({
            "x": np.asarray(las.x),
            "y": np.asarray(las.y),
            "z": np.asarray(las.z),
        })
        veg_df["gtid"] = pd.NA

        # Merge by nearest XYZ (within small tolerance)
        # We assume segmentation.xyz has same coordinates; exact match join is fine.
        logging.debug(f"[{tile_dir.name}] Merging vegetation with segmentation by XYZ.")
        merged = pd.merge(
            veg_df, seg_df[["x", "y", "z", "gtid"]],
            on=["x", "y", "z"], how="inner"
        )
        if merged.empty:
            logging.debug(f"[{tile_dir.name}] No coordinate matches — skipped.")
            continue

        # Write new LAS with GTID attribute
        try:
            logging.debug(f"Adding gtid field to vegetation LAS: {veg_path}")

            header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
            las_out = laspy.LasData(header)
            las_out.x = merged["x"].values
            las_out.y = merged["y"].values
            las_out.z = merged["z"].values

            # Add new extra dimension 'gtid'
            if "gtid" not in las_out.point_format.extra_dimension_names:
                las_out.add_extra_dim(laspy.ExtraBytesParams(name="gtid", type=np.int64))
            las_out.gtid = merged["gtid"].astype("int64").values

            las_out.write(out_forest)
            logging.info(f"[{tile_dir.name}] Wrote forest.laz ({len(las_out.points)} points)")
        except Exception as e:
            logging.warning(f"[{tile_dir.name}] Failed writing forest.laz: {e}")

    return {
        "case": case,
        "status": "ok",
        "n_trees": n_trees,
        "outputs": {
            "forest_hulls": out_forest_hulls,
            "gtid_map": out_gtid_map,
        },
    }
