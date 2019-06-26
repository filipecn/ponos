enable_language("CUDA")

add_executable(poseidon_tests 
main.cpp 
fd.cu
smoke_solver.cu
particle_system.cu
level_set.cu
)

target_link_libraries(poseidon_tests gtest pthread poseidon hermes )
set_target_properties(poseidon PROPERTIES
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_STANDARD 11
)
target_include_directories(poseidon_tests PUBLIC ${POSEIDON_SOURCE_DIR} ${HERMES_SOURCE_DIR})