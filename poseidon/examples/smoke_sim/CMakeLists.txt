add_executable(ss2 ss2.cpp)
target_compile_definitions(ss2 PUBLIC -DSHADERS_PATH="")
target_include_directories(ss2 PUBLIC
        ${POSEIDON_SOURCE_DIR} ${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR})
target_link_libraries(ss2 PRIVATE poseidon ponos circe)
