enable_language("CUDA")
set(UTILS_SRCS render.h render.cu)
# 2d
add_executable(cuSmoke2 main.cu ${UTILS_SRCS})
set_target_properties(cuSmoke2 PROPERTIES
LINKER_LANGUAGE CUDA
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CUDA_STANDARD 11
)
target_include_directories(cuSmoke2 PUBLIC 
${LODEPNG_INCLUDES} 
${POSEIDON_SOURCE_DIR} ${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR} ${HERMES_SOURCE_DIR} 
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuSmoke2 PRIVATE hermes poseidon ponos circe lodepng cufft)
# 2d
add_executable(cuSmoke2d main2.cu ${UTILS_SRCS})
set_target_properties(cuSmoke2d PROPERTIES
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CUDA_STANDARD 11
)
target_include_directories(cuSmoke2d PUBLIC 
${LODEPNG_INCLUDES} 
${POSEIDON_SOURCE_DIR} ${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR} ${HERMES_SOURCE_DIR} 
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuSmoke2d PRIVATE hermes poseidon ponos circe lodepng cufft)
# 3d
add_executable(cuSmoke3d main3.cu ${UTILS_SRCS})
set_target_properties(cuSmoke3d PROPERTIES
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CUDA_STANDARD 11
)
target_include_directories(cuSmoke3d PUBLIC 
${LODEPNG_INCLUDES} 
${POSEIDON_SOURCE_DIR} ${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR} ${HERMES_SOURCE_DIR} 
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuSmoke3d PRIVATE hermes poseidon ponos circe lodepng cufft)