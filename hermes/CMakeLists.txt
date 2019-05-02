project(HERMES CUDA)

set(HERMES_HEADERS
        hermes/hermes.h
        hermes/colors/cuda_color.h
        hermes/colors/cuda_color_palette.h
        hermes/common/cuda.h
        hermes/common/cuda_parallel.h
        hermes/common/cuda_random.h
        hermes/common/defs.h
        hermes/geometry/cuda_bbox.h
        hermes/geometry/cuda_bbox.inl
        hermes/geometry/cuda_matrix.h
        hermes/geometry/cuda_numeric.h
        hermes/geometry/cuda_transform.h
        hermes/geometry/cuda_transform.inl
        hermes/geometry/cuda_vector.h
        hermes/geometry/cuda_vector.inl
        hermes/geometry/cuda_point.h
        hermes/geometry/cuda_point.inl
        hermes/numeric/cuda_field.h
        hermes/numeric/cuda_vector_field.h
        hermes/numeric/cuda_grid.h
        hermes/numeric/cuda_staggered_grid.h
        hermes/parallel/cuda_reduce.h
        hermes/storage/cuda_array.h
        hermes/storage/cuda_array_kernels.h
        hermes/storage/cuda_texture.h
        hermes/storage/cuda_texture.inl
        hermes/storage/cuda_texture_kernels.h
        )
        
set(HERMES_SOURCES
        hermes/storage/cuda_texture_kernels.cu
        )

enable_language("CUDA")
add_library(hermes STATIC ${HERMES_SOURCES} ${HERMES_HEADERS})
set_target_properties(hermes PROPERTIES
LINKER_LANGUAGE CUDA
CUDA_STANDARD 11
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CMAKE_CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME "hermes"
        FOLDER "HERMES")
target_include_directories(hermes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory(tests)
