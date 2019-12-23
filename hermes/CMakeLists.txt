project(HERMES CUDA)

set(HERMES_HEADERS
        hermes/hermes.h
        hermes/algorithms/cuda_marching_cubes.h
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
        hermes/geometry/cuda_point.h
        hermes/geometry/cuda_point.inl
        hermes/numeric/cuda_field.h
        hermes/numeric/cuda_interpolation.h
        hermes/numeric/cuda_vector_field.h
        hermes/numeric/cuda_vector_grid.cu
        hermes/numeric/cuda_grid.h
        hermes/numeric/cuda_grid_kernels.h
        hermes/numeric/cuda_numeric.h
        hermes/numeric/cuda_staggered_grid.h
        hermes/parallel/cuda_reduce.h
        hermes/storage/cuda_array.h
        hermes/storage/cuda_texture.h
        hermes/storage/cuda_texture.inl
        hermes/storage/cuda_texture_kernels.h
        hermes/storage/cuda_storage_utils.h
        )
        
set(HERMES_SOURCES
        hermes/algorithms/cuda_marching_cubes.cu
        hermes/numeric/cuda_grid.cu
        hermes/numeric/cuda_interpolation.cu
        hermes/storage/cuda_texture_kernels.cu
        )

# find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

enable_language("CUDA")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#CUDA_COMPILE_PTX(ptx_files ${HERMES_SOURCES} ${HERMES_HEADERS})
add_library(hermes STATIC ${HERMES_SOURCES} ${HERMES_HEADERS})# ${ptx_files})
set_target_properties(hermes PROPERTIES
LINKER_LANGUAGE CUDA
CUDA_STANDARD 11
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CMAKE_CUDA_SEPARABLE_COMPILATION ON
POSITION_INDEPENDENT_CODE OFF
        OUTPUT_NAME "hermes"
        FOLDER "HERMES")
target_include_directories(hermes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )

if(BUILD_TESTS)
        add_subdirectory(tests)
endif(BUILD_TESTS)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hermes
        DESTINATION ${INSTALL_PATH}/include
        FILES_MATCHING REGEX "(.*\\.[inl|h])")

install(TARGETS hermes
        LIBRARY DESTINATION ${INSTALL_PATH}/lib
        ARCHIVE DESTINATION ${INSTALL_PATH}//lib
        )

if(BUILD_SHARED)
	add_library(hermes_shared SHARED ${HERMES_SOURCES} ${HERMES_HEADERS})
	set_target_properties(hermes_shared PROPERTIES
		LINKER_LANGUAGE CUDA
		CUDA_STANDARD 11
		CUDA_RESOLVE_DEVICE_SYMBOLS ON
		CMAKE_CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME "hermes_shared"
        FOLDER "HERMES_SHARED")
	target_include_directories(hermes_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )
	install(TARGETS hermes hermes_shared
        LIBRARY DESTINATION ${INSTALL_PATH}/lib
        ARCHIVE DESTINATION ${INSTALL_PATH}//lib
        )
endif(BUILD_SHARED)