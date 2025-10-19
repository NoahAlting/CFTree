#!/usr/bin/env python3
# geometry/write_cityjson.py

# Standard library
import os, gc, json, uuid, argparse, subprocess, tempfile, shutil
from pathlib import Path
from time import perf_counter as now

# Limit threading to 1 core for numerical libs to avoid memory explosion explicitly
for thread_env_var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TBB_NUM_THREADS"]:
    os.environ[thread_env_var] = os.environ.get(thread_env_var, "1")

# Scientific stack
import numpy as np
import laspy
import trimesh
from scipy.spatial import cKDTree, ConvexHull
import rasterio
from rasterio import mask
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shp_transform
from trimesh.geometry import align_vectors


# ================ logging functions ================
def log_step(gtid, label, secs=None, **fields):
    head = f"[{gtid}]"
    lab  = f"{label:<10}"                       # left-align to width 10
    tpart = f"{secs:>7.3f}s" if secs is not None else " " * 8
    kv = "  ".join(f"{k}={_format_val(k, v)}" for k, v in fields.items())
    print(f"{head:<6} {lab} {tpart}  {kv}")

def _format_val(k, v):
    """Pretty formatting for values."""
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if k in ("r50", "porosity"):
            return f"{v:.5f}"
        if k in ("volume", "h", "CW", "crown_median_z", "H", "DBH", "r_trunk"):
            return f"{v:.3f}"
        return f"{v:.3f}"
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)

def log_total(secs):
    print(f"{'':>20}TIME total={secs:.3f}s")


# ================ point cloud utilities ================
def build_gtid_index(las, id_field="gtid"):
    # get ID column
    try:
        ids = getattr(las, id_field)
    except AttributeError:
        ids = las[id_field]

    # normalize to a hashable key array
    if np.issubdtype(ids.dtype, np.number):
        keys = ids.astype(np.int64)
    elif ids.dtype.kind in "SU":
        keys = ids.astype(str)
    else:
        keys = np.array([
            (x.decode("utf-8", "ignore").strip() if isinstance(x, (bytes, bytearray)) else str(x))
            for x in ids
        ], dtype=object)

    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    # start positions of new keys
    starts = np.r_[0, np.flatnonzero(keys_sorted[1:] != keys_sorted[:-1]) + 1]
    ends = np.r_[starts[1:], keys_sorted.size]

    idx = {}
    for s, e in zip(starts, ends):
        k = str(keys_sorted[s])
        idx[k] = order[s:e]
    return idx

def voxel_downsample(points_xyz: np.ndarray, voxel: float) -> np.ndarray:
    if points_xyz.size == 0 or voxel <= 0: return points_xyz
    bmin = points_xyz.min(axis=0)
    ijk = np.floor((points_xyz - bmin) / voxel).astype(np.int64)
    _, keep_idx = np.unique(ijk, axis=0, return_index=True)
    keep_idx.sort()
    return points_xyz[keep_idx]

def point_voxel_keys(points_xyz: np.ndarray, origin: np.ndarray, voxel: float):
    ijk = np.floor((points_xyz - origin) / voxel).astype(np.int64, copy=False)
    ijk = np.ascontiguousarray(ijk)
    idx_dtype = np.dtype([("i", np.int64), ("j", np.int64), ("k", np.int64)])
    keys = ijk.view(idx_dtype).reshape(-1)
    uniq, _ = np.unique(keys, return_index=True)
    return np.frombuffer(uniq.tobytes(), dtype=np.int64).reshape(-1, 3)


# ================ metrics ================
def single_grid_porosity(mesh, pts_xyz, h, log=False, label=None, return_stats=False):

    bmin, bmax = mesh.bounds
    origin = bmin  # deterministic zero-phase

    # grid axes
    nx = int(np.ceil((bmax[0] - origin[0]) / h))
    ny = int(np.ceil((bmax[1] - origin[1]) / h))
    nz = int(np.ceil((bmax[2] - origin[2]) / h))
    if nx <= 0 or ny <= 0 or nz <= 0:
        return float("nan")

    xs = origin[0] + (np.arange(nx) + 0.5) * h
    ys = origin[1] + (np.arange(ny) + 0.5) * h
    zs = origin[2] + (np.arange(nz) + 0.5) * h

    # interior centers on grid
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
    centers = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    inside = mesh.contains(centers)
    interior_mask = inside.reshape(ny, nx, nz, order="C")
    interior_count = int(interior_mask.sum())

    if interior_count == 0:
        if log:
            tag = f"[{label}] " if label else ""
            print(f"{tag} voxels={nx*ny*nz:,} interior=0 (h={h:.4f})")
        return float("nan")

    # unique occupied voxel keys from points
    occ_keys = point_voxel_keys(pts_xyz, origin=origin, voxel=h)  # (M,3) int indices

    # vectorized tally of occupied voxels that are also interior
    i, j, k = occ_keys[:, 0], occ_keys[:, 1], occ_keys[:, 2]
    valid = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
    occ_interior = int(interior_mask[j[valid], i[valid], k[valid]].sum())

    vox_total = nx * ny * nz
    porosity = (interior_count - occ_interior) / interior_count
    if return_stats:
        return float(porosity), {"voxels": vox_total, "interior": interior_count}
    return float(porosity)

def nn_median_r50(points_xyz: np.ndarray, mesh: trimesh.Trimesh,
                  nn_samples=500_000, down_voxel=0.02, batch=100000) -> float:
    """
    Compute median nearest-neighbor distance (r50) between interior voxels
    of the crown mesh and the point cloud.
    pitch is derived from bbox_vol/nn_samples
    """
    # Interior pool (deterministic phase)
    bmin, bmax = mesh.bounds
    span = (bmax - bmin)
    bbox_vol = float(np.prod(span))
    pitch = max((bbox_vol / max(nn_samples, 1)) ** (1/3), 1e-3)

    vg = mesh.voxelized(pitch).fill()
    P = vg.points
    if P.shape[0] == 0:
        raise RuntimeError("No interior voxels for r50.")

    take = min(nn_samples, P.shape[0])
    inside = P[:take]
    origin = mesh.bounds[0]

    # Downsample points
    pts_ds = voxel_downsample(points_xyz, down_voxel) if down_voxel > 0 else points_xyz
    pts_local = np.asarray(pts_ds - origin, dtype=np.float32, order="C")
    inside_local = np.asarray(inside - origin, dtype=np.float32, order="C")

    # KDTree nearest-neighbor distances, in batches
    tree = cKDTree(pts_local, compact_nodes=True, balanced_tree=True)
    d = np.empty(inside_local.shape[0], dtype=np.float32)
    for i in range(0, inside_local.shape[0], batch):
        j = min(i + batch, inside_local.shape[0])
        d[i:j], _ = tree.query(inside_local[i:j], k=1, workers=1)

    r50 = float(np.median(d))
    return r50


# ================ tree attributes ================
def estimate_dbh_from_crown(cw_m: float, h_m: float, a=1.0, b=1.1, c=0.7) -> float:
    """
    Estimate DBH (in meters) from crown width and height (in meters)
    using an allometric relation of the form:
    DBH(cm) = a * (CW[m])^b * (H[m])^c
    The result is converted from centimeters to meters by dividing by 100.
    """
    if not np.isfinite(cw_m) or cw_m <= 0 or not np.isfinite(h_m) or h_m <= 0:
        return np.nan
    return float(a * (cw_m ** b) * (h_m ** c) / 100.0) #cast to meters

def apply_slenderness_floor(dbh_m: float, h_m: float, s_max: float = 120.0) -> float:
    """
    Enforce H/DBH ≤ s_max. 
    If DBH is too small for the given height, lift it to DBH_min = H(m) / s_max.
    Input/output: meters.
    """
    if not np.isfinite(dbh_m) or dbh_m <= 0 or not np.isfinite(h_m) or h_m <= 0:
        return dbh_m
    dbh_min_m = h_m / float(s_max)
    return max(float(dbh_m), float(dbh_min_m))


# ================ Geometry helpers ================
def compute_alpha_wrap_points(alpha_wrapper_path:str, pts_xyz: np.ndarray, ralpha: float = 10.0, roffset: float = 10.0,
                              tmpdir: str | None = None) -> trimesh.Trimesh:
    """
    Run CGAL Alpha_wrap_3 on a point set and return a trimesh.Trimesh.
    Uses a temp .xyz input and .ply output; cleans up after.
    alpha = diag/ralpha, offset = alpha/roffset  (matches the CLI you compiled)
    """
    if pts_xyz.ndim != 2 or pts_xyz.shape[1] != 3:
        raise ValueError("pts_xyz must be (N,3) array")

    if tmpdir is None:
        # prefer RAM disk if available
        td = "/dev/shm"
        tmpdir = td if (os.path.isdir(td) and os.access(td, os.W_OK)) else tempfile.gettempdir()

    stem = f"aw_{uuid.uuid4().hex}"
    in_xyz = os.path.join(tmpdir, stem + ".xyz")
    out_ply = os.path.join(tmpdir, stem + ".ply")

    # write XYZ (fast; no headers)
    np.savetxt(in_xyz, pts_xyz, fmt="%.8f")

    try:
        if not (alpha_wrapper_path and os.path.isfile(alpha_wrapper_path)):
            raise RuntimeError(f"awrap binary not found: {alpha_wrapper_path!r}")
        # call CGAL wrapper
        sp = subprocess.run([alpha_wrapper_path, in_xyz, str(ralpha), str(roffset), out_ply],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # load mesh into memory
        mesh = trimesh.load(out_ply, force="mesh", skip_materials=True, process=False)
        if isinstance(mesh, trimesh.Scene):
            geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(geoms) if geoms else trimesh.Trimesh()
        return mesh
    finally:
        for p in (in_xyz, out_ply):
            try: os.remove(p)
            except OSError: pass

def compute_crown_metrics(crown_mesh: trimesh.Trimesh):
    """
    Compute crown metrics:
      - Crown width (CW) from 2D convex hull (fallback = max XY span of bbox).
      - Median crown height (Z).
    Returns (CW_m, crown_median_z).
    """
    xy = crown_mesh.vertices[:, :2]
    if xy.shape[0] < 3:
        CW = float("nan")
    else:
        try:
            hull = ConvexHull(xy)
            area = float(hull.volume)  # in 2D: 'volume' is area
            CW = 2.0 * np.sqrt(area / np.pi)  # equivalent-circle diameter
        except Exception:
            bmin, bmax = crown_mesh.bounds
            CW = float(max(bmax[0] - bmin[0], bmax[1] - bmin[1]))

    crown_median_z = float(np.median(crown_mesh.vertices[:, 2]))
    return CW, crown_median_z

def compute_trunk_base_from_dtm(crown_mesh, dtm_path):
    """Return trunk base (bx, by, bz) or None if no valid DTM support.
    closest dtm pixel to centroid so that slanted crowns over e.g. water can still have trunk base"""

    # Crown footprint polygon
    hull = ConvexHull(crown_mesh.vertices[:, :2])
    poly = Polygon(crown_mesh.vertices[hull.vertices, :2])
    crown_centroid = crown_mesh.centroid[:2]

    with rasterio.open(dtm_path) as src:
        # Clip the DTM to the crown footprint
        out_img, out_transform = mask.mask(src, [poly], crop=True, filled=False)
        band = out_img[0]

        rows, cols = np.where(~band.mask)
        if len(rows) == 0:
            return None  # no valid DTM pixels under crown footprint

        # Convert row/col to x,y using the affine transform
        xs, ys = rasterio.transform.xy(out_transform, rows, cols)
        coords = np.column_stack([xs, ys])
        vals = band.data[rows, cols]

        # Pick the closest pixel to the crown centroid
        dists = np.linalg.norm(coords - crown_centroid, axis=1)
        idx = np.argmin(dists)

        bx, by = coords[idx]
        bz = vals[idx]
        return np.array([bx, by, bz], dtype=float)


def build_lod0_geometry(crown_mesh, trunk_base, city):
    """LoD0: canopy footprint polygon at DTM height."""
    if trunk_base is None:
        return None

    # Project crown mesh XY and take convex hull
    pts_xy = crown_mesh.vertices[:, :2]
    hull = ConvexHull(pts_xy)
    poly = pts_xy[hull.vertices]

    z0 = float(trunk_base[2])
    V = np.column_stack([poly, np.full(len(poly), z0)])
    vbase = len(city["vertices"])
    city["vertices"].extend(V.tolist())

    ring = (np.arange(len(poly)) + vbase).tolist()
    return {"type": "MultiSurface", "lod": 0.0, "boundaries": [[ring]]}

def build_lod1_geometry(crown_mesh, trunk_base, crown_median_z, city):
    """LoD1: extruded footprint from DTM to crown median height."""
    if trunk_base is None:
        return None

    pts_xy = crown_mesh.vertices[:, :2]
    hull = ConvexHull(pts_xy)
    poly = pts_xy[hull.vertices]

    z0 = float(trunk_base[2])
    z1 = float(crown_median_z)

    V_bottom = np.column_stack([poly, np.full(len(poly), z0)])
    V_top = np.column_stack([poly, np.full(len(poly), z1)])
    V = np.vstack([V_bottom, V_top])

    vbase = len(city["vertices"])
    city["vertices"].extend(V.tolist())

    n = len(poly)
    # Bottom face, top face
    bottom = list(range(vbase, vbase + n))
    top = list(range(vbase + n, vbase + 2*n))

    # Walls: connect each bottom/top edge
    walls = []
    for i in range(n):
        j = (i + 1) % n
        walls.append([bottom[i], bottom[j], top[j], top[i]])

    shell = [[bottom], [top], *[[w] for w in walls]]
    return {"type": "Solid", "lod": 1.0, "boundaries": [shell]}

def build_lod2_crown(CW, crown_median_z, centroid_xy, city):
    """LoD2 crown as a sphere with diameter = CW, centered on crown median height."""
    if not np.isfinite(CW) or CW <= 0:
        return None

    radius = CW * 0.5
    center = np.array([centroid_xy[0], centroid_xy[1], crown_median_z], dtype=float)

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere.apply_translation(center)

    V = sphere.vertices.astype(float)
    F = sphere.faces.astype(int)
    vbase = len(city["vertices"])
    city["vertices"].extend(V.tolist())

    boundaries = [[(f + vbase).tolist()] for f in F]
    return {"type": "Solid", "lod": 2.0, "boundaries": [boundaries]}

def build_lod2_trunk(trunk_base, crown_median_z, r_trunk, city):
    """LoD2 trunk as a straight vertical cylinder."""
    if trunk_base is None or not np.isfinite(r_trunk) or r_trunk <= 0:
        return None, None

    height = crown_median_z - trunk_base[2]
    if height <= 0:
        return None, None

    cyl = trimesh.creation.cylinder(radius=r_trunk, height=height, sections=16)
    cyl.apply_translation([trunk_base[0], trunk_base[1], trunk_base[2]])

    Vc = cyl.vertices.astype(float)
    Fc = cyl.faces.astype(int)
    vbase_c = len(city["vertices"])
    city["vertices"].extend(Vc.tolist())

    shell = [[(f + vbase_c).tolist()] for f in Fc]
    return {"type": "Solid", "lod": 2.0, "boundaries": [shell]}, height

def build_lod3_crown(crown_mesh, city):
    """Add crown mesh as MultiSurface geometry (LoD3)."""
    V = crown_mesh.vertices.astype(float)
    F = crown_mesh.faces.astype(int)
    vbase = len(city["vertices"])
    city["vertices"].extend(V.tolist())

    # Each face is one polygon surface (single ring)
    boundaries = [[(f + vbase).tolist()] for f in F]
    return {"type": "MultiSurface", "lod": 3.0, "boundaries": boundaries}

def build_lod3_trunk(crown_mesh, trunk_base, r_trunk, crown_median_z, city):
    """
    Build a slanted cylinder for the trunk with horizontal caps,
    add its vertices to `city["vertices"]`, and return a CityJSON Solid geometry (LoD3)
    plus the 3D trunk length.

    Returns:
        trunk_geom (dict or None), trunk_length (float or None)
    """
    # Axis: from base to crown "top" point (median Z at crown centroid XY)
    top = np.array([crown_mesh.centroid[0], crown_mesh.centroid[1], crown_median_z], dtype=float)
    base = trunk_base.astype(float)
    axis = top - base
    trunk_length = float(np.linalg.norm(axis))

    if not np.isfinite(trunk_length) or trunk_length <= 0:
        return None, None

    # Cylinder along +Z of length = trunk_length, radius = r_trunk
    cyl = trimesh.creation.cylinder(radius=r_trunk, height=trunk_length, sections=32)

    # Move cylinder so its base sits at Z=0 (horizontal bottom cap), then rotate to axis, then translate to base
    cyl.apply_translation([0.0, 0.0, -cyl.bounds[0, 2]])
    R = align_vectors([0, 0, 1], axis / trunk_length)
    cyl.apply_transform(R)
    cyl.apply_translation(base)

    # Push vertices; build Solid boundaries (one exterior shell)
    Vc = cyl.vertices.astype(float)
    Fc = cyl.faces.astype(int)
    vbase_c = len(city["vertices"])
    city["vertices"].extend(Vc.tolist())

    # CityJSON Solid requires: boundaries = [ shell ], and each surface is a list of rings.
    shell = [[(f + vbase_c).tolist()] for f in Fc]          # list of triangle surfaces (each with 1 ring)
    solid_boundaries = [shell]                                # one exterior shell

    trunk_geom = {"type": "Solid", "lod": 3.0, "boundaries": solid_boundaries}
    return trunk_geom, trunk_length


# ================ CityJSON assembly ================
def add_tree_cityobject(city, obj_id, geoms, attrs):
    """Add a tree as a SolitaryVegetationObject with given geometries + attributes."""
    city["CityObjects"][obj_id] = {
        "type": "SolitaryVegetationObject",
        "geometry": geoms,
        "attributes": dict(attrs),
    }

def add_tree_cityobject_child_parent(city, obj_id, geoms, attrs, trunk_valid):
    """Assemble a tree as parent + child CityObjects."""

    # parent node (no geometry, only attributes and children)
    children_ids = []
    if "crown" in geoms and geoms["crown"] is not None:
        cid = f"{obj_id}_crown"
        children_ids.append(cid)
        city["CityObjects"][cid] = {
            "type": "GenericCityObject",
            "geometry": [geoms["crown"]],
            "attributes": {"role": "crown"},
        }
    if trunk_valid and "trunk" in geoms and geoms["trunk"] is not None:
        tid = f"{obj_id}_trunk"
        children_ids.append(tid)
        city["CityObjects"][tid] = {
            "type": "GenericCityObject",
            "geometry": [geoms["trunk"]],
            "attributes": {"role": "trunk"},
        }

    # parent tree
    city["CityObjects"][obj_id] = {
        "type": "SolitaryVegetationObject",
        "children": children_ids,
        "attributes": dict(attrs),
    }

def finalize_cityjson_metadata(city: dict, crs: str = "EPSG:28992") -> dict:
    """
    Finalize CityJSON dict with transform, bounding box, CRS, and LoDs.
    Ensures output is CJ 2.0-compliant and ready for merging.
    """
    V = np.asarray(city["vertices"], dtype=float)
    if V.size == 0:
        return city

    # Bounding box
    bmin = V.min(axis=0)
    bmax = V.max(axis=0)
    bbox = [float(x) for x in [bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]]]

    # Transform (integerized to mm precision)
    scale = [0.001, 0.001, 0.001]
    translate = [float(bmin[0]), float(bmin[1]), float(bmin[2])]
    V_int = np.round((V - translate) / scale).astype(int)

    # Update city object
    city["type"] = "CityJSON"
    city["version"] = "2.0"
    city["vertices"] = V_int.tolist()
    city["transform"] = {"scale": scale, "translate": translate}
    city["metadata"] = {
        "referenceSystem": f"https://www.opengis.net/def/crs/{crs}",
        "geographicalExtent": bbox,
        "presentLoDs": [1.0, 2.0]
    }
    return city


# ================ Main script ================
def warmup_once():
    """Warm up SciPy cKDTree, trimesh ray 'contains', and voxel engine (single core)."""

    # 1) KDTree warmup
    pts = np.random.rand(2000, 3).astype(np.float32)
    q   = np.random.rand(200,  3).astype(np.float32)
    cKDTree(pts, compact_nodes=True, balanced_tree=True).query(q, k=1, workers=1)

    # 2) Trimesh ray 'contains' warmup
    verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=int)
    m = trimesh.Trimesh(verts, faces, process=False)
    _ = m.contains(np.array([[0.1,0.1,0.1],[2,2,2]], dtype=float))

    # 3) Voxelization warmup
    _ = m.voxelized(0.2).fill()

    try:
        print("[init] embree:", trimesh.ray.has_embree)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--las", default="forest.laz")
    ap.add_argument("--dtm_path", default="clipped_dtm.tif", help="Path to DTM GeoTIFF")
    ap.add_argument("--limit", type=int, default=None, help="export first N GTIDs")
    ap.add_argument('--alpha_wrapper_path', type=str, default=os.path.expanduser("~/geometry_builder/awrap/build/awrap_points"), help="path to alpha wrap executable")
    ap.add_argument("--ralpha", type=float, default=15)
    ap.add_argument("--roffset", type=float, default=50)
    ap.add_argument("--k", type=float, default=0.80, help="voxel pitch h = k * r50")
    ap.add_argument("--h-min", type=float, default=0.02)
    ap.add_argument("--h-max", type=float, default=1.0)
    ap.add_argument("--nn-samples", type=int, default=600_000)
    ap.add_argument("--down-voxel", type=float, default=0.00,
                    help="for ALS no downsampling needed as typical trees are <20k points")
    ap.add_argument("--out", default=os.path.join("results", "cityjson2", "trees_final.city.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    las = laspy.read(args.las)
    gtid_idx = build_gtid_index(las, id_field='gtid')
    all_gtids = sorted(gtid_idx.keys(), key=str)
    gtids = all_gtids[:args.limit] if args.limit else all_gtids

    print(f"[init] GTIDs in index: {len(gtid_idx)}")
    missing = [g for g in gtids if g not in gtid_idx]
    if missing:
        print(f"[init][warn] {len(missing)} GTIDs not found in LAS (e.g., {missing[:5]})")

    city = {"CityObjects": {}, "vertices": []}

    print(f"[init] warming up voxel engine")
    warmup_once()
    print(f"[init] all warmed up, lets go!\n" + "-" * 50)

    X, Y, Z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)

    for i, gtid in enumerate(gtids):
        try:
            t0 = now()
            print("-" * 50 + f"[cityjson] ({i+1}/{len(gtids)})")

            # 1) collect points
            t_pts0 = now()
            idx = gtid_idx.get(gtid)
            if idx is None or len(idx) == 0:
                raise RuntimeError(f"No points for GTID {gtid}")
            pts = np.column_stack((X[idx], Y[idx], Z[idx])).astype(np.float64, copy=False)
            log_step(gtid, "pts", now() - t_pts0, N=pts.shape[0])

            # 2) alpha wrap mesh
            t_wrap0 = now()
            crown_wrap_mesh = compute_alpha_wrap_points(args.alpha_wrapper_path, pts, ralpha=args.ralpha, roffset=args.roffset)
            if crown_wrap_mesh.is_empty or crown_wrap_mesh.vertices.size == 0 or crown_wrap_mesh.faces.size == 0:
                print(f"[cityjson][WARN] {gtid}: empty wrap; skipping")
                continue
            log_step(gtid, "wrap", now() - t_wrap0,
                    mesh_v=int(crown_wrap_mesh.vertices.shape[0]),
                    f=int(crown_wrap_mesh.faces.shape[0]),
                    volume=float(crown_wrap_mesh.volume),
                    watertight=bool(crown_wrap_mesh.is_watertight))

            # 3) r50
            t_r500 = now()
            r50 = nn_median_r50(pts, crown_wrap_mesh,
                                nn_samples=args.nn_samples,
                                down_voxel=args.down_voxel)
            h = float(np.clip(args.k * r50, args.h_min, args.h_max))
            log_step(gtid, "r50", now() - t_r500, r50=float(r50), h=h)

            # 4) porosity
            t_por0 = now()
            por, stats = single_grid_porosity(crown_wrap_mesh, pts, h, return_stats=True)
            log_step(gtid, "vox_grid", now() - t_por0, **{k: int(v) for k, v in stats.items()})
            log_step(gtid, "por", now() - t_por0, porosity=float(por))

            # 5) crown metrics
            t_crown0 = now()
            CW, crown_median_z = compute_crown_metrics(crown_wrap_mesh)
            log_step(gtid, "crown", now() - t_crown0, CW=float(CW), crown_median_z=float(crown_median_z))

            # 6) trunk base
            t_trunk0 = now()
            trunk_base = compute_trunk_base_from_dtm(crown_wrap_mesh, args.dtm_path)
            H, DBH, r_trunk, trunk_geom, trunk_length, trunk_valid, trunk_tilt = (None, None, None, None, None, False, None)

            if trunk_base is not None:
                H = float(crown_median_z - trunk_base[2])
                if np.isfinite(H) and H > 0:
                    DBH = estimate_dbh_from_crown(CW, H)
                    # DBH = apply_slenderness_floor(DBH, H)
                    r_trunk = float(DBH * 0.5)
                    if np.isfinite(r_trunk) and r_trunk > 0:
                        trunk_geom, trunk_length = build_lod3_trunk(crown_wrap_mesh, trunk_base, r_trunk, crown_median_z, city)
                        trunk_valid = trunk_geom is not None
                if trunk_valid:
                    # horizontal offset from crown centroid to trunk base
                    vertical_span = crown_median_z - trunk_base[2]
                    horiz_offset = np.hypot(*(crown_wrap_mesh.centroid[:2] - trunk_base[:2]))
                    trunk_tilt = np.degrees(np.arctan2(horiz_offset, vertical_span))
            else:
                print(f"[cityjson][WARN] {gtid}: trunk_base not found in DTM")

            log_step(gtid, "trunk", now() - t_trunk0, H=H, DBH=DBH, r_trunk=r_trunk, trunk_valid=trunk_valid)

            # 7) assemble attributes and geometries
            t_add0 = now()
            attrs = {
                "gtid": gtid,
                "N_points": int(pts.shape[0]),
                "model_classification": "porous_crown",
                "porosity": float(por),
                "r50_m": float(r50),
                "DBH_m": float(DBH) if np.isfinite(DBH) else None,
                "crown_eq_circle_width_m": float(CW) if CW is not None else None,
                "crown_median_z_m": float(crown_median_z) if np.isfinite(crown_median_z) else None,
                "trunk_valid": bool(trunk_valid),
                "trunk_radius_m": float(r_trunk) if trunk_valid else None,
                "trunk_length_m": float(trunk_length) if trunk_valid else None,
                "trunk_tilt_deg": float(trunk_tilt) if trunk_tilt is not None else None,
                "dtm_height_m": float(trunk_base[2]) if trunk_base is not None else None,
            }

            
            # # LoD0
            # g0 = build_lod0_geometry(crown_wrap_mesh, trunk_base, city)
            # if g0: geoms.append(g0)

            # # LoD1
            # g1 = build_lod1_geometry(crown_wrap_mesh, trunk_base, crown_median_z, city)
            # if g1: geoms.append(g1)

            # # LoD2 (analytic sphere crown + straight trunk)
            # g2c = build_lod2_crown(CW, crown_median_z, crown_wrap_mesh.centroid[:2], city)
            # g2t, _ = build_lod2_trunk(trunk_base, crown_median_z, r_trunk, city)
            # if g2c: geoms.append(g2c)
            # if g2t: geoms.append(g2t)

            ###########
            # LoD3 (mesh crown + slanted trunk)
            # g3c = build_lod3_crown(crown_wrap_mesh, city)
            # if g3c: geoms.append(g3c)
            # if trunk_valid and trunk_geom: geoms.append(trunk_geom)
            ###########
            child_parent_format = False

            if child_parent_format:
                geoms = {}
                g3c = build_lod3_crown(crown_wrap_mesh, city)
                if g3c: geoms["crown"] = g3c
                if trunk_valid and trunk_geom: geoms["trunk"] = trunk_geom

                # 8) add CityObject
                add_tree_cityobject_child_parent(city, f"tree_{gtid}", geoms, attrs, trunk_valid)

            else:
                geoms = []
                # LoD3 (mesh crown + slanted trunk)
                g3c = build_lod3_crown(crown_wrap_mesh, city)
                if g3c: geoms.append(g3c)
                if trunk_valid and trunk_geom: geoms.append(trunk_geom)
                
                # 8) add CityObject
                add_tree_cityobject(city, f"tree_{gtid}", geoms, attrs)

            # 8) add CityObject
            # add_tree_cityobject(city, f"tree_{gtid}", geoms, attrs)


            log_step(gtid, "geometry", now() - t_add0)
            log_total(now() - t0)

            del crown_wrap_mesh, pts
            gc.collect()

        except Exception as e:
            print(f"[cityjson][WARN] {gtid}: {e}")
            continue


    city = finalize_cityjson_metadata(city)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(city, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[cityjson] wrote {out_path}")



if __name__ == "__main__":
    main()

