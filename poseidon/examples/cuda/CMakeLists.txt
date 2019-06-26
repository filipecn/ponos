
enable_language("CUDA")
# LEVEL SET 2
add_executable(level_set2 level_set2.cu render.cu)
target_compile_definitions(level_set2 PUBLIC -DSHADERS_PATH="${POSEIDON_SOURCE_DIR}/examples/shaders")
set_target_properties(level_set2 PROPERTIES
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CUDA_STANDARD 11
)
target_include_directories(level_set2 PUBLIC 
${LODEPNG_INCLUDES} 
${POSEIDON_SOURCE_DIR} ${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR} ${HERMES_SOURCE_DIR} 
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(level_set2 PRIVATE hermes poseidon ponos circe lodepng cufft)
# LEVEL SET 3
add_executable(level_set3 level_set3.cu render.cu)
target_compile_definitions(level_set3 PUBLIC -DSHADERS_PATH="${POSEIDON_SOURCE_DIR}/examples/shaders")
set_target_properties(level_set3 PROPERTIES
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CUDA_STANDARD 11
)
target_include_directories(level_set3 PUBLIC 
${LODEPNG_INCLUDES} 
${POSEIDON_SOURCE_DIR} ${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR} ${HERMES_SOURCE_DIR} 
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(level_set3 PRIVATE hermes poseidon ponos circe lodepng cufft)
# PRACTICAL LIQUIDS 2
add_executable(practical_liquids2 practical_liquids2.cu render.cu)
target_compile_definitions(practical_liquids2 PUBLIC -DSHADERS_PATH="${POSEIDON_SOURCE_DIR}/examples/shaders")
set_target_properties(practical_liquids2 PROPERTIES
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CUDA_STANDARD 11
)
target_include_directories(practical_liquids2 PUBLIC 
${LODEPNG_INCLUDES} 
${POSEIDON_SOURCE_DIR} ${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR} ${HERMES_SOURCE_DIR} 
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(practical_liquids2 PRIVATE hermes poseidon ponos circe lodepng cufft)



add_subdirectory(smoke2d)