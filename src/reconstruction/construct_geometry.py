# src/reconstruction/construct_geometry.py
"""
Construct LoD3 tree geometries (crown + trunk) in local coordinates.

Both crown and trunk are represented as CityJSON "Solid" geometries.
This module is pure: no file I/O or side effects beyond logging.

Inputs:
    - crown_mesh : trimesh.Trimesh       # alpha-wrapped crown in local coords
    - metrics     : dict                 # from extract_tree_metrics
    - offset_global : list[float]        # translation back to RD New
    - gtid, tile_id : optional identifiers for logging

Outputs:
    dict with:
        {
            "components": [
                {"role": "crown", "lod": 3.0, "vertices_local": np.ndarray, "faces": np.ndarray},
                {"role": "trunk", "lod": 3.0, "vertices_local": np.ndarray, "faces": np.ndarray},
            ],
            "attributes": OrderedDict([...])
        }
"""

from __future__ import annotations
import logging
from collections import OrderedDict
from typing import Optional
import numpy as np
import trimesh


# ---------------------------------------------------------------------
# Canonical attribute order (extendable later)
# ---------------------------------------------------------------------
ATTR_KEYS = [
    "gtid", "tile_id",
    "crown_width_m", "crown_median_z", "crown_r50_m", "crown_porosity",
    "trunk_H_m", "trunk_DBH_m", "trunk_radius_m",
    "trunk_base_height_m"
]


# ---------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------
def _build_crown_solid(mesh: trimesh.Trimesh, gtid: Optional[int] = None) -> Optional[dict]:
    """Return crown as a Solid (LoD3)."""
    if mesh.is_empty or mesh.vertices.size == 0 or mesh.faces.size == 0:
        logging.debug(f"[GTID {gtid}] Crown mesh empty — skipped.")
        return None

    vertices_local = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    logging.debug(
        f"[GTID {gtid}] Crown Solid: {len(vertices_local)} verts, {len(faces)} faces, "
        f"volume={mesh.volume:.3f}, watertight={mesh.is_watertight}"
    )

    return {
        "role": "crown",
        "lod": 3.0,
        "vertices_local": vertices_local,
        "faces": faces,
    }


def _build_trunk_solid(
    crown_mesh: trimesh.Trimesh,
    trunk_base: np.ndarray,
    r_trunk: float,
    crown_median_z: float,
    gtid: Optional[int] = None,
) -> Optional[dict]:
    """Return slanted trunk as a Solid (LoD3)."""
    try:
        if trunk_base is None or not np.all(np.isfinite(trunk_base)):
            logging.debug(f"[GTID {gtid}] Trunk base invalid — skipped.")
            return None
        if not np.isfinite(r_trunk) or r_trunk <= 0:
            logging.debug(f"[GTID {gtid}] Invalid trunk radius ({r_trunk}) — skipped.")
            return None

        # Axis from base → (crown centroid XY, crown_median_z)
        top = np.array([crown_mesh.centroid[0], crown_mesh.centroid[1], crown_median_z], dtype=float)
        base = np.asarray(trunk_base, dtype=float)
        axis = top - base
        length = np.linalg.norm(axis)
        if not np.isfinite(length) or length <= 0:
            logging.debug(f"[GTID {gtid}] Invalid trunk length ({length}) — skipped.")
            return None

        # Create cylinder aligned along +Z, bottom at (0, 0, 0)
        cyl = trimesh.creation.cylinder(radius=r_trunk, height=length, sections=24)
        cyl.apply_translation([0, 0, length / 2])

        # Align +Z axis with direction from base to top
        R = trimesh.geometry.align_vectors([0, 0, 1], axis / length)
        cyl.apply_transform(R)
        cyl.apply_translation(base)

        logging.debug(
            f"[GTID {gtid}] Trunk Solid: r={r_trunk:.3f} m, length={length:.3f} m, "
            f"{len(cyl.vertices)} verts, {len(cyl.faces)} faces"
        )

        return {
            "role": "trunk",
            "lod": 3.0,
            "vertices_local": np.asarray(cyl.vertices, dtype=float),
            "faces": np.asarray(cyl.faces, dtype=int),
        }

    except Exception as e:
        logging.debug(f"[GTID {gtid}] Trunk Solid construction failed: {e}")
        return None


# ---------------------------------------------------------------------
# Attribute normalization
# ---------------------------------------------------------------------
def _normalize_attributes(metrics: dict, gtid: int, tile_id: Optional[str] = None) -> OrderedDict:
    """
    Flatten and order attributes to a stable OrderedDict with None for missing.
    """
    crown = metrics.get("crown", {})
    trunk = metrics.get("trunk", {})

    vals = {
        "gtid": gtid,
        "tile_id": tile_id,
        "crown_width_m": crown.get("CW_m"),
        "crown_median_z": crown.get("median_z"),   
        "crown_r50_m": crown.get("r50_m"),
        "crown_porosity": crown.get("porosity"),
        "trunk_H_m": trunk.get("H_m"),
        "trunk_DBH_m": trunk.get("DBH_m"),
        "trunk_radius_m": trunk.get("r_trunk"),
    }
    base = trunk.get("base_xyz")                         

    if base is not None and len(base) == 3:
        vals.update({"trunk_base_height_m": base[2]})
    else:
        vals.update({"trunk_base_height_m": None})

    ordered = OrderedDict()
    for k in ATTR_KEYS:
        v = vals.get(k, None)
        if isinstance(v, float) and not np.isfinite(v):
            v = None
        ordered[k] = v

    # Compact, aligned one-line summary for debug readability
    vals_fmt = ", ".join(
        f"{k.split('.')[-1]}={v:.3f}" if isinstance(v, (float, int)) and v is not None else f"{k.split('.')[-1]}={v}"
        for k, v in ordered.items()
    )
    logging.debug(f"[GTID {gtid}] Normalized attributes: {vals_fmt}")

    return ordered


# ---------------------------------------------------------------------
# Main LoD3 constructor
# ---------------------------------------------------------------------
def construct_lod3(
    crown_mesh: trimesh.Trimesh,
    metrics: dict,
    offset_global: list[float] | np.ndarray,
    gtid: Optional[int] = None,
    tile_id: Optional[str] = None,
) -> dict:
    """
    Construct LoD3 geometries (crown + trunk) for one tree in local coordinates.

    Returns dict with:
        {"components": [ ... ], "attributes": OrderedDict([...])}
    """
    gtid_str = f"GTID {gtid}" if gtid is not None else "GTID ?"
    logging.debug(f"[{tile_id}] [{gtid_str}] Constructing LoD3 geometry...")

    components = []

    # --- Crown Solid (local coordinates)---
    crown_comp = _build_crown_solid(crown_mesh, gtid)
    if crown_comp is not None:
        components.append(crown_comp)
    else:
        logging.warning(f"[{tile_id}] [{gtid_str}] No valid crown component created.")

    # --- Trunk Solid ---
    trunk_base = None
    try:
        trunk_base_global = np.array(metrics["trunk"]["base_xyz"], dtype=float)
        trunk_base_local = trunk_base_global - np.asarray(offset_global, dtype=float)
    except Exception:
        trunk_base_local = None

    crown_median_z_global = float(metrics.get("crown", {}).get("median_z", np.nan))
    crown_median_z_local = crown_median_z_global - float(np.asarray(offset_global, dtype=float)[2])

    # --- Build trunk in LOCAL coordinates ---
    trunk_comp = _build_trunk_solid(
        crown_mesh=crown_mesh,
        trunk_base=trunk_base_local,
        r_trunk=float(metrics.get("trunk", {}).get("r_trunk", np.nan)),
        crown_median_z=crown_median_z_local,
        gtid=gtid
    )


    if trunk_comp is not None:
        components.append(trunk_comp)
    else:
        logging.warning(f"[{tile_id}] [{gtid_str}] No valid trunk component created.")

    # --- Sanity check: local coordinates shouldn't exceed ~5 000 m magnitude ---
    for comp in components:
        vmax = np.abs(np.asarray(comp.get("vertices_local", []))).max() if comp.get("vertices_local") is not None else 0
        if np.isfinite(vmax) and vmax > 5000:
            logging.warning(
                f"[Tile {tile_id} | GTID {gtid}] Suspicious local magnitude ({vmax:.1f}); "
                f"possible global coords leaked into local component."
            )

    # --- Attributes ---
    attributes = _normalize_attributes(metrics, gtid or -1, tile_id)

    logging.info(
        f"[{tile_id}] [{gtid_str}] Constructed {len(components)} LoD3 components "
        f"(Crown={any(c['role']=='crown' for c in components)}, "
        f"Trunk={any(c['role']=='trunk' for c in components)})"
    )

    return {"components": components, "attributes": attributes}
