import os
import sys
import subprocess
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from shapely.geometry import MultiPoint
from concurrent.futures import ProcessPoolExecutor

# --------------------------
# Input arguments
# --------------------------
if len(sys.argv) < 2:
    print("Usage: python segmentation.py <city> [cores]")
    sys.exit(1)

code_dir = os.path.join('/home', 'npalting', 'cftree', 'code')
data_dir = os.path.join('/data2', 'npalting', sys.argv[1])
tiles_dir = os.path.join(data_dir, 'tiles')

cores = int(sys.argv[2]) if len(sys.argv) > 2 else 8

# --------------------------
# segment trees in a tile
# --------------------------
def segment_tile(tile_path):
    vegetation_xyz = os.path.join(tile_path, "vegetation.xyz")
    segmentation_xyz = os.path.join(tile_path, "segmentation.xyz")
    tree_hulls_geojson = os.path.join(tile_path, "tree_hulls.geojson")

    if not os.path.exists(vegetation_xyz):
        print(f"[segment_tile] Skipping: missing input {vegetation_xyz}", flush=True)
        return

    segmentation_exe = os.path.join(code_dir, 'segmentation_code', 'build', 'segmentation')
    segmentation_params = {
        'radius': 2.5,
        'vres': 1.5,
        'min_pts': 3
    }

    print(f"Running segmentation on {vegetation_xyz}", flush=True)
    cmd = [
        segmentation_exe,
        vegetation_xyz,
        segmentation_xyz,
        str(segmentation_params["radius"]),
        str(segmentation_params["vres"]),
        str(segmentation_params["min_pts"])
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Segmentation failed for {tile_path}: {e}", flush=True)
        return

    seg_df = pd.read_csv(segmentation_xyz, sep=r"\s+", header=None, names=["tid", "x", "y", "z"])
    seg_gdf = gpd.GeoDataFrame(seg_df, geometry=gpd.points_from_xy(seg_df.x, seg_df.y), crs="EPSG:28992")

    hulls = []
    for tid, group in seg_gdf.groupby("tid"):
        if len(group) >= 3:
            hull_geom = MultiPoint(group.geometry.values).convex_hull
            hulls.append({"tid": tid, "geometry": hull_geom})
        else:
            print(f"tid {tid} has fewer than 3 points — skipped")

    hulls_gdf = gpd.GeoDataFrame(hulls, crs="EPSG:28992")
    hulls_gdf.to_file(tree_hulls_geojson, driver="GeoJSON")
    print(f"Segmentation complete: {tile_path}", flush=True)

# --------------------------
# Parallel execution
# --------------------------
if __name__ == "__main__":
    tile_folders = [os.path.join(tiles_dir, f) for f in os.listdir(tiles_dir)
                    if os.path.isdir(os.path.join(tiles_dir, f))]

    with ProcessPoolExecutor(max_workers=cores) as executor:
        list(tqdm(executor.map(segment_tile, tile_folders), total=len(tile_folders), desc="segmenting tiles"))
