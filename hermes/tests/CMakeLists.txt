set(SOURCES 
    main.cu
    numeric_tests.cu
    memory_tests.cu
)

add_executable(hermes_tests ${SOURCES})
target_include_directories(hermes_tests PUBLIC ${CATCH2_INCLUDES})
target_link_libraries(hermes_tests ponos hermes)

target_compile_options(hermes_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       -arch=sm_50
                        -Xcompiler -pg
                       --relocatable-device-code=true
                    #    -–extended-lambda
                    >)

set_target_properties(hermes_tests PROPERTIES
        LINKER_LANGUAGE CUDA
        CMAKE_CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 11
        CMAKE_CUDA_STANDARD_REQUIRED ON
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        )
