"""
src/reconstruction/alpha_wrap_tree.py

Python interface for per-tree alpha wrapping using the CGAL CLI binary.

Wraps: src/reconstruction/AlphaWrap/build/awrap_points

Reads:
    <cache_dir>/tree_<gtid>.xyz

Writes:
    <cache_dir>/tree_<gtid>.ply   # temporary geometry (deleted later)

Returns:
    {
        "gtid": str,
        "status": "ok" | "failed" | "missing_input" | "missing_binary",
        "outputs": {"mesh_ply": Path}
    }
"""

from __future__ import annotations
import subprocess
import logging
from pathlib import Path
from typing import Optional


def alpha_wrap_tree(
    tree_xyz: Path,
    cache_dir: Path,
    ralpha: float = 15.0,
    roffset: float = 50.0,
    binary_path: Optional[Path] = None,
    overwrite: bool = False
) -> dict:
    """
    Run CGAL alpha wrapping on a single tree point cloud.

    Parameters
    ----------
    tree_xyz : Path
        Input .xyz file containing tree points.
    cache_dir : Path
        Directory for temporary files (_cache/).
    ralpha : float, default=15.0
        Alpha scaling factor relative to point cloud diagonal.
    roffset : float, default=50.0
        Offset scaling factor relative to alpha.
    binary_path : Path, optional
        Path to compiled awrap_points binary.
    overwrite : bool, default=False
        If True, re-run even if output already exists.
    """
    gtid = tree_xyz.stem.split("_")[-1]
    mesh_ply = cache_dir / f"tree_{gtid}.ply"
    binary_path = binary_path or Path(__file__).parent / "AlphaWrap" / "build" / "awrap_points"

    # Pre-checks
    if not tree_xyz.exists():
        logging.warning(f"[GTID {gtid}] Missing input file: {tree_xyz}")
        return {"gtid": gtid, "status": "missing_input", "outputs": {}}

    if not binary_path.exists():
        logging.error(f"[GTID {gtid}] Missing alpha wrap binary: {binary_path}")
        return {"gtid": gtid, "status": "missing_binary", "outputs": {}}

    if mesh_ply.exists() and not overwrite:
        logging.debug(f"[GTID {gtid}] Existing mesh found, skipping.")
        return {"gtid": gtid, "status": "skipped", "outputs": {"mesh_ply": mesh_ply}}

    # Run binary
    cmd = [
        str(binary_path),
        str(tree_xyz),
        str(ralpha),
        str(roffset),
        str(mesh_ply)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.debug(f"[GTID {gtid}] Alpha wrap complete -> {mesh_ply.name}")
        return {"gtid": gtid, "status": "ok", "outputs": {"mesh_ply": mesh_ply}}
    except subprocess.CalledProcessError as e:
        logging.warning(f"[GTID {gtid}] Alpha wrapping failed: {e.stderr.decode(errors='ignore')}")
        return {"gtid": gtid, "status": "failed", "outputs": {}}
    except Exception as e:
        logging.error(f"[GTID {gtid}] Unexpected error: {e}")
        return {"gtid": gtid, "status": "failed", "outputs": {}}
