cmake_minimum_required(VERSION 3.10)
project(OpticalFlow3D)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ITK REQUIRED)
find_package(VTK REQUIRED)
include(${ITK_USE_FILE})
include(${VTK_USE_FILE})

add_executable(optical_flow_3d FLux3D.cpp)
target_link_libraries(optical_flow_3d ${ITK_LIBRARIES} ${VTK_LIBRARIES})

