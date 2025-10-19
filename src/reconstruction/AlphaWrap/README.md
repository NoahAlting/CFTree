# AlphaWrap — CGAL Alpha Wrapping for CFTree

This module provides a minimal, reproducible **CGAL Alpha Wrapping** executable (`awrap_points`)
used by the CFTree Step 3 geometry reconstruction pipeline.

It is a lightly adapted version of the official CGAL example  
[`Alpha_wrap_3/point_set_wrap.cpp`](https://doc.cgal.org/latest/Alpha_wrap_3/Alpha_wrap_3_2point_set_wrap_8cpp-example.html),
configured for batch execution from Python and CFD-oriented tree reconstruction.

---

## 🧩 Overview

The program wraps a 3D point cloud (`.xyz`) into a **watertight surface mesh** using  
[`CGAL::alpha_wrap_3`](https://doc.cgal.org/latest/Alpha_wrap_3/index.html).  
It automatically scales the wrapping parameters relative to the point set’s bounding-box diagonal:

\[ 
\alpha = \frac{\text{diag}}{r_\alpha}, \quad \text{offset} = \frac{\alpha}{r_\text{offset}}
\]

The resulting mesh is written in **binary PLY** format, suitable for direct loading by
[`trimesh`](https://trimsh.org/) in Python.

---

## ⚙️ Build Instructions

CMake ≥ 3.18 and a C++17 compiler are required.

```bash
cd src/reconstruction/AlphaWrap
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## dependencies
| Library                                | Notes                       |
| -------------------------------------- | --------------------------- |
| [CGAL](https://www.cgal.org/) ≥ 5.5    | Required for `alpha_wrap_3` |
| [Eigen3](https://eigen.tuxfamily.org/) | Linear algebra backend      |
| [Boost](https://www.boost.org/)        | Utility headers             |
| CMake, Make, g++ / clang++             | Build toolchain             |

If using Conda:
``` bash
conda install -c conda-forge cgal boost-cpp eigen cmake make gxx_linux-64
```

## usage
Run directly from terminal:
``` bash
./awrap_points <input.xyz> [ralpha=15] [roffset=50] [output.ply|-]
```

Examples:
``` bash 
./awrap_points tree_points.xyz 15 50 crown_mesh.ply
# or pipe binary PLY to stdout:
./awrap_points tree_points.xyz 15 50 - > crown_mesh.ply
```


## 🔬 Testing (optional)

You can verify the binary independently before integrating with Python:

```bash
# Generate a random point cloud (example)
python - <<'EOF'
import numpy as np
np.savetxt("sample.xyz", np.random.rand(1000, 3) * 10)
EOF

# Run alpha wrapping
./awrap_points sample.xyz 15 50 sample_wrap.ply
```
The file sample_wrap.ply should now contain a watertight triangular mesh that can be viewed in Meshlab or loaded via trimesh.load("sample_wrap.ply").