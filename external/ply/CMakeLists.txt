project(PLY)

cmake_minimum_required(VERSION 2.8.7)

add_library(ply ply.h plyfile.c)
target_include_directories(ply PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(ply PROPERTIES
        OUTPUT_NAME "ply"
        FOLDER "PLY")
