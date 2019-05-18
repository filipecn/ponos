enable_language("CUDA")

add_executable(hermes_tests 
main.cpp 
#particle_system.cu 
#array.cu
utils.h
utils.cu
transform.cu
texture.cu
field.cu
grids.cu
memory.cu
blas.cu
)

target_link_libraries(hermes_tests gtest pthread hermes )
set_target_properties(hermes PROPERTIES
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_STANDARD 11
)
target_include_directories(hermes_tests PUBLIC ${HERMES_SOURCE_DIR})