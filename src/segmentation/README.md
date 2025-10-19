# TreeSeparation Integration

This segmentation module integrates the **TreeSeparation** C++ code by Jinhu Wang  
<https://github.com/Jinhu-Wang/TreeSeparation>, licensed under **LGPL-3.0**.

## Build Instructions
```bash
cd src/segmentation/TreeSeparation
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```