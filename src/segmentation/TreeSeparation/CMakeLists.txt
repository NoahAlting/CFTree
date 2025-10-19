cmake_minimum_required(VERSION 3.10)
project(segmentation_code)

# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)


# Add the executable. Adjust the source file list if needed.
add_executable(segmentation Source.cpp FoxTree.cpp)

