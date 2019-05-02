enable_language("CUDA")
add_executable(cuSmoke2d main.cu
render.h
render.cu
)
set_target_properties(cuSmoke2d PROPERTIES
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_STANDARD 11
)
target_include_directories(cuSmoke2d PUBLIC 
    ${LODEPNG_INCLUDES} 
${POSEIDON_SOURCE_DIR} ${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR} ${HERMES_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuSmoke2d PRIVATE hermes poseidon ponos circe lodepng cufft)