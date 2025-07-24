cmake_minimum_required(VERSION 3.10)
project(B-spline)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ITK REQUIRED)
find_package(VTK REQUIRED)
include(${ITK_USE_FILE})
include(${VTK_USE_FILE})

add_executable(B-spline B-spline.cpp)
target_link_libraries(B-spline ${ITK_LIBRARIES} ${VTK_LIBRARIES})