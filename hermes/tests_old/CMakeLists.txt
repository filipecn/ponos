enable_language("CUDA")

set(sources
        main.cpp
        #particle_system.cu
        #array.cu
        #utils.h
        #algorithms.cu
        #utils.cu
        #transform.cu
        numeric.cu
        #texture.cu
        #field.cu
        #grids.cu
        memory.cu
        #blas.cu
        )

#include_directories(${HERMES_SOURCE_DIR} "${gtest_SOURCE_DIR}/include")
#CUDA_COMPILE_PTX(ptx_files ${sources})
add_executable(hermes_tests ${sources})
target_compile_options(hermes_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       -gencode arch=compute_35,code=sm_35
                       --relocatable-device-code=true
                    #    -–extended-lambda
                    >)
target_compile_options(hermes_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)
# target_link_options(hermes_tests PUBLIC
# LINKER_LANGUAGE CUDA
# )

set_target_properties(hermes_tests PROPERTIES
        LINKER_LANGUAGE CUDA
        CMAKE_CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 11
CMAKE_CUDA_STANDARD_REQUIRED ON
        CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
        )
target_include_directories(hermes_tests PUBLIC ${HERMES_SOURCE_DIR})

if (WIN32)
    target_link_libraries(hermes_tests hermes gtest)
elseif (APPLE)
    target_link_libraries(hermes_tests hermes gtest pthread)
else (WIN32)
    target_link_libraries(hermes_tests hermes gtest gcov pthread)
endif (WIN32)

