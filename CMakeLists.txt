cmake_minimum_required(VERSION 3.0.0)
project(cgmres-cpp)

set(CGMRES_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(example/semiactive_damper)
add_subdirectory(test)
