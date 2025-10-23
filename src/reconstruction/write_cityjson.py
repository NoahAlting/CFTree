# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/reconstruction/write_cityjson.py

"""
Assemble and finalize CityJSON tiles for tree geometries (LoD3).

This module is stateless and pure:
- No file I/O (runner writes final dict to JSON file).
- Only operates on in-memory CityJSON dicts.
- Logs progress and structure for reproducibility.

Structure per tree:
    parent  -> SolitaryVegetationObject
        ├── child "_crown"  -> GenericCityObject (Solid LoD3)
        └── child "_trunk"  -> GenericCityObject (Solid LoD3)
"""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import trimesh

# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------
def init_cityjson() -> dict:
    """
    Initialize an empty CityJSON structure.
    """
    city = {
        "type": "CityJSON",
        "version": "2.0",
        "CityObjects": {},
        "vertices": [],
    }
    logging.info("[cityjson] Initialized new CityJSON structure")
    return city

def format_crs_uri(crs_str: str) -> str:
    """
    Convert a CRS code like 'EPSG:28992' into a valid OGC CRS URI for CityJSON.
    Example: 'EPSG:28992' -> 'https://www.opengis.net/def/crs/EPSG/0/28992'
    """
    if not crs_str:
        return None
    if crs_str.upper().startswith("EPSG:"):
        epsg_code = crs_str.split(":")[1]
        return f"https://www.opengis.net/def/crs/EPSG/0/{epsg_code}"
    
    return crs_str

# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def _append_vertices(city: dict, verts_global: np.ndarray) -> tuple[list[list[float]], int]:
    vbase = len(city["vertices"])
    n = len(verts_global)
    city["vertices"].extend(verts_global.tolist())
    logging.debug(
        f"[cityjson] Appended {n} vertices (vbase={vbase}, range={vbase}–{vbase+n-1})"
    )
    return verts_global.tolist(), vbase



def _solid_from_faces(faces: np.ndarray, base_index: int) -> dict:
    """
    Create a valid CityJSON Solid geometry.
    Expected structure:
      Solid → [shell]
      shell → [surface]
      surface → [ring]
      ring → [vertex_indices]
    So: [[[ [v1, v2, v3] ], [ [v4, v5, v6] ] ]]
    """
    # Correct 4-level nesting
    shell = [[(f + base_index).tolist()] for f in faces]

    solid = {
        "type": "Solid",
        "lod": 3.0,
        "boundaries": [shell],
    }

    preview = ", ".join(str(f.tolist()) for f in faces[:3])
    logging.debug(
        f"[cityjson] Solid built with {len(faces)} faces "
        f"(vbase={base_index}, first_faces={preview})"
    )

    return solid


def _compute_bbox(city: dict) -> list[float]:
    """Compute correct bbox [xmin, ymin, zmin, xmax, ymax, zmax]."""
    verts = np.asarray(city["vertices"], dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3:
        logging.warning("[cityjson] Invalid vertex array shape for bbox computation.")
        return [0, 0, 0, 0, 0, 0]
    mins = np.nanmin(verts, axis=0)
    maxs = np.nanmax(verts, axis=0)
    bbox = [float(mins[0]), float(mins[1]), float(mins[2]),
            float(maxs[0]), float(maxs[1]), float(maxs[2])]
    logging.debug(f"[cityjson] Bounding box computed: {bbox}")
    return bbox




# ---------------------------------------------------------------------
# Add tree to CityJSON
# ---------------------------------------------------------------------
def add_tree(
    city: dict,
    gtid: int,
    components: list[dict],
    offset_global: list[float] | np.ndarray,
    attributes: dict,
) -> None:
    """
    Add a SolitaryVegetationObject (tree) with multiple LoD3 geometries (crown/trunk)
    to the CityJSON structure. Each geometry is a Solid; all attributes belong
    to the tree object itself (no parent/child structure).
    """
    try:
        offset = np.asarray(offset_global, dtype=float)
    except Exception:
        offset = np.zeros(3, dtype=float)

    obj_id = f"T_{gtid}"
    geometries = []

    logging.debug(f"[GTID {gtid}] ----- Constructing CityObject {obj_id}")

    for comp in components:
        role = comp.get("role", "unknown")
        verts_local = np.asarray(comp["vertices_local"], dtype=float)
        faces = np.asarray(comp["faces"], dtype=int)

        logging.debug(
            f"[GTID {gtid}] {role}: verts_local={verts_local.shape}, "
            f"faces={faces.shape}, offset={offset.tolist()}"
        )

        verts_global = verts_local + offset
        _, vbase = _append_vertices(city, verts_global)

        solid = _solid_from_faces(faces, vbase)
        solid["lod"] = comp.get("lod", 3.0)

        geometries.append(solid)

        logging.debug(
            f"[GTID {gtid}] Added {role} geometry: "
            f"{len(verts_local)} verts, {len(faces)} faces, vbase={vbase}"
        )

    city["CityObjects"][obj_id] = {
        "type": "SolitaryVegetationObject",
        "geometry": geometries,
        "attributes": dict(attributes),
    }

    logging.debug(
        f"[GTID {gtid}] Final CityObject has {len(geometries)} geometries, "
        f"total vertices={len(city['vertices'])}"
    )

# ---------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------
def finalize_cityjson(city: dict, crs: str | None = None) -> dict:
    """
    Finalize CityJSON by quantizing vertices, adding transform and metadata.
    """
    if crs is None:
        from src.config import get_config
        crs = get_config()["crs"]

    if not city["vertices"]:
        logging.warning("[cityjson] No vertices to finalize — returning empty CityJSON.")
        return city

    # --- 1. Compute bounding box in real-world coords ---
    V = np.asarray(city["vertices"], dtype=float)
    if V.ndim != 2 or V.shape[1] != 3:
        logging.error(f"[cityjson] Invalid vertex array shape: {V.shape}")
        return city

    bmin = np.nanmin(V, axis=0)
    bmax = np.nanmax(V, axis=0)
    bbox_real = [float(x) for x in [bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]]]

    # --- 2. Quantization (millimeter precision) ---
    scale = [0.001, 0.001, 0.001]
    translate = [float(bmin[0]), float(bmin[1]), float(bmin[2])]
    V_int = np.round((V - translate) / scale).astype(int)

    city["vertices"] = V_int.tolist()
    city["transform"] = {"scale": scale, "translate": translate}

    # --- 3. Write metadata with real-world extent ---
    city["metadata"] = {
        "referenceSystem": format_crs_uri(crs),
        "geographicalExtent": bbox_real,
        "presentLoDs": [3.0],
    }

    # --- 4. Logging ---
    dx, dy, dz = (bmax - bmin)
    logging.info(
        f"[cityjson] Finalized with {len(city['CityObjects'])} objects, "
        f"{len(V_int)} vertices, "
        f"bbox=({bmin[0]:.1f},{bmin[1]:.1f},{bmin[2]:.1f})–"
        f"({bmax[0]:.1f},{bmax[1]:.1f},{bmax[2]:.1f}), "
        f"extent≈({dx:.1f}×{dy:.1f}×{dz:.1f}) m"
    )

    return city
