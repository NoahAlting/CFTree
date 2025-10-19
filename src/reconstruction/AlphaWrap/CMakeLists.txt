cmake_minimum_required(VERSION 3.18)
project(AlphaWrapPoints LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CGAL REQUIRED COMPONENTS Core)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

add_executable(awrap_points point_set_wrap_cli.cpp)
target_link_libraries(awrap_points PRIVATE CGAL::CGAL Boost::boost)
target_include_directories(awrap_points PRIVATE ${EIGEN3_INCLUDE_DIR})
