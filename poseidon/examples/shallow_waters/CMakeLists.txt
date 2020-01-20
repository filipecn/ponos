add_executable(sw1 sw1.cpp)
target_compile_definitions(sw1 PUBLIC -DSHADERS_PATH="")
target_include_directories(sw1 PUBLIC 
${LODEPNG_INCLUDES} 
${POSEIDON_SOURCE_DIR} ${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR} ${HERMES_SOURCE_DIR} 
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(sw1 PRIVATE hermes poseidon ponos circe )
