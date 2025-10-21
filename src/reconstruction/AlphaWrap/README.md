# AlphaWrap Integration

This module integrates and adapts the **CGAL Alpha Wrapping** example  
[`Alpha_wrap_3/point_set_wrap.cpp`](https://doc.cgal.org/latest/Alpha_wrap_3/Alpha_wrap_3_2point_set_wrap_8cpp-example.html)  
from the [CGAL library](https://www.cgal.org/), used in the **CFTree** pipeline for watertight 3D tree reconstruction.

## Overview
Alpha Wrapping creates a **watertight triangular mesh** from a 3D point cloud using  
[`CGAL::alpha_wrap_3`](https://doc.cgal.org/latest/Alpha_wrap_3/index.html).  
It guarantees manifold, closed surfaces suitable for CFD applications and is used here to reconstruct tree crowns.

## Modifications
This version is a light adaptation of the official CGAL example, modified to:
- Support non-interactive batch execution from Python.
- Adjust input/output handling for `.xyz` point clouds.
- Output binary PLY files directly readable by `trimesh`.
- Simplify console output and error handling.

The geometric and algorithmic behavior of the original example remains unchanged.

## Build Instructions
Requires **CMake ≥ 3.18** and a **C++17-compatible compiler** (`g++` or `clang++`).

```bash
cd src/reconstruction/AlphaWrap
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This will compile the executable awrap_points inside the `build/` directory.
Ensure this binary is built and accessible before running `scripts/tree_reconstruction.py`.

## Dependencies
| Library                                | Purpose                     |
| -------------------------------------- | --------------------------- |
| [CGAL](https://www.cgal.org/) ≥ 5.5    | Core Alpha Wrapping library |
| CMake, Make, g++ / clang++             | Build toolchain             |

If using Conda:
``` bash
conda install -c conda-forge cgal boost-cpp eigen cmake make gxx_linux-64
```

## License Notice
This directory contains a derivative of the CGAL Alpha Wrap 3 example,
which is released under the GPL-3.0 license as part of CGAL.