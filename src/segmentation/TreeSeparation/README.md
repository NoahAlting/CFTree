# TreeSeparation Integration

This module integrates and adapts the **TreeSeparation** C++ implementation by  
[Jinhu Wang](https://github.com/Jinhu-Wang/TreeSeparation), originally licensed under **LGPL-3.0**.

## Overview
TreeSeparation performs individual tree segmentation from airborne LiDAR point clouds.  
In this project, it is used as part of the **HOMED + TreeSeparation** segmentation pipeline for urban trees.

## Modifications
The source code has been slightly modified to:
- Adjust input/output handling for use within the Python pipeline.
- Streamline console output and error handling.
- Enable compatibility with AHN5 tile naming and coordinate conventions.

These changes do not alter the underlying algorithmic logic of TreeSeparation.

## Build Instructions
A C++17-compatible compiler (e.g., `g++` ≥ 9) and CMake (≥ 3.10) are required.

```bash
cd src/segmentation/TreeSeparation
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Licence Notice
This directory contains derivative work under the LGPL-3.0 license.
Original copyright © 2021 Jinhu Wang.
