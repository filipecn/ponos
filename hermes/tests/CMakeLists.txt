enable_language("CUDA")

set(sources
        main.cpp
        #particle_system.cu
        #array.cu
        utils.h
        algorithms.cu
        utils.cu
        transform.cu
        numeric.cu
        texture.cu
        field.cu
        grids.cu
        memory.cu
        blas.cu
        )

#include_directories(${HERMES_SOURCE_DIR} "${gtest_SOURCE_DIR}/include")
#CUDA_COMPILE_PTX(ptx_files ${sources})
add_executable(hermes_tests ${sources})

if (WIN32)
    target_link_libraries(hermes_tests hermes gtest)
elseif (APPLE)
    target_link_libraries(hermes_tests hermes gtest pthread)
else (WIN32)
    target_link_libraries(hermes_tests hermes gtest gcov pthread)
endif (WIN32)

set_target_properties(hermes PROPERTIES
        CMAKE_CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 11
        )
target_include_directories(hermes_tests PUBLIC ${HERMES_SOURCE_DIR})