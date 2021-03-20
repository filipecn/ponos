project(PONOS)

set(PONOS_HEADERS
        ponos/ponos.h
        #ponos/algorithm/fast_sweep.h
        ponos/algorithm/geometric_shapes.h
        #ponos/algorithm/marching_squares.h
        ponos/algorithm/scatter.h
        ponos/algorithm/sort.h
        ponos/algorithm/triangulate.h
        #ponos/blas/bicg.h
        ponos/blas/blas.h
        #ponos/blas/cg.h
        ponos/blas/c_regular_grid.h
        ponos/blas/c_regular_grid.inl
        #ponos/blas/custom_matrix.h
        #ponos/blas/fdm_cg_solver.h
        #ponos/blas/fdm_gauss_seidel_solver.h
        #ponos/blas/fdm_jacobi_solver.h
        #ponos/blas/fdm_matrix.h
        #ponos/blas/fdm_vector.h
        ponos/blas/field_grid.h
        ponos/blas/field_grid.inl
        ponos/blas/field.h
        ponos/blas/vector.h
        #ponos/blas/gauss_jordan.h
        #ponos/blas/interpolator.h
        #ponos/blas/interpolator.inl
        #ponos/blas/kernels.h
        #ponos/blas/linear_system.h
        #ponos/blas/pcg.h
        #ponos/blas/rbf.h
        #ponos/blas/solver_interface.h
        #ponos/blas/sparse_cg_solver.h
        #ponos/blas/sparse_gauss_seidel_solver.h
        #ponos/blas/sparse_jacobi_solver.h
        #ponos/blas/sparse_matrix.h
        #ponos/blas/sparse_sor_solver.h
        #ponos/blas/sparse_vector.h
        #ponos/blas/staggered_grid.h
        #ponos/blas/staggered_grid.inl
        #ponos/blas/symmetric_matrix.h
		ponos/common/arg_parser.h
		ponos/common/bitmask_operators.h
        ponos/common/defs.h
        ponos/common/index.h
        ponos/common/file.h
		ponos/common/file_system.h
		ponos/storage/array_of_structures.h
		ponos/storage/memory.h
        ponos/storage/array.h
        ponos/common/parallel.h
        ponos/random/noise.h
		ponos/random/random.h
		ponos/common/str.h
        ponos/common/timer.h
		ponos/geometry/math_element.h
        ponos/geometry/bbox.h
        ponos/geometry/ellipse.h
        ponos/geometry/frustum.h
		ponos/numeric/differential_operators.h
        ponos/numeric/interpolation.h
        ponos/numeric/fd_matrix.h
		ponos/numeric/grid_interpolator.h
        ponos/geometry/interval.h
        ponos/geometry/interval.inl
        ponos/geometry/line.h
        ponos/geometry/matrix.h
        ponos/geometry/normal.h
        ponos/numeric/numeric.h
        ponos/geometry/parametric_surface.h
        ponos/geometry/plane.h
        ponos/geometry/point.h
	ponos/geometry/point.inl
        ponos/geometry/polygon.h
        ponos/geometry/quaternion.h
        ponos/geometry/queries.h
        ponos/geometry/ray.h
        ponos/geometry/segment.h
        ponos/geometry/shape.h
        ponos/geometry/sphere.h
        ponos/geometry/surface.h
        ponos/geometry/transform.h
        ponos/geometry/utils.h
        ponos/geometry/vector.h
        ponos/geometry/vector.inl
        ponos/log/debug.h
		ponos/log/memory_dump.h
		ponos/log/console_colors.h
		ponos/parallel/parallel.h
        ponos/numeric/grid.h
        ponos/numeric/vector_grid.h
        ponos/storage/memory_block.h
        ponos/structures/bah.h
        ponos/structures/brep.hpp
        ponos/structures/bvh.h
        ponos/numeric/grid_interface.h
        #ponos/structures/half_edge3.h
        #ponos/structures/half_edge.h
        ponos/numeric/level_set.h
        ponos/structures/mesh.h
        ponos/structures/object_pool.h
		ponos/structures/n_mesh.h
		ponos/structures/p_mesh.h
        ponos/structures/octree.h
        ponos/structures/octree.inl
        ponos/structures/point_set_interface.h
        ponos/structures/quad_tree.h
        ponos/structures/quad_tree.inl
        ponos/structures/raw_mesh.h
        ponos/structures/tetrahedron_mesh.h
        ponos/structures/tetrahedron_mesh.inl
        ponos/structures/z_grid.h
        ponos/structures/z_point_set.h
        )

file(GLOB PONOS_SOURCES
        ponos/algorithm/*.cpp
        ponos/geometry/*.cpp
        ponos/blas/*.cpp
		ponos/log/*.cpp
		ponos/parallel/*.cpp
		ponos/random/*.cpp
		ponos/common/*.cpp
        ponos/structures/*.cpp
		ponos/storage/*.cpp
		ponos/numeric/*.cpp
        )

add_library(ponos STATIC ${PONOS_SOURCES} ${PONOS_HEADERS})
set_target_properties(ponos PROPERTIES
        OUTPUT_NAME "ponos"
        FOLDER "PONOS")
target_include_directories(ponos PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
		${SPDLOG_INCLUDE_DIR}
		)
target_link_libraries(ponos
		${SPDLOG_LIBRARIES}
		${ponos_LIBRARIES}
		${PONOS_LIBRARIES})
add_dependencies(ponos spdlog)

if(BUILD_EXAMPLES)
 #       add_subdirectory(examples)
endif(BUILD_EXAMPLES)
if(BUILD_TESTS)
        add_subdirectory(tests)
endif(BUILD_TESTS)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ponos
        DESTINATION ${INSTALL_PATH}/include
        FILES_MATCHING REGEX "(.*\\.[inl|h])")

install(TARGETS ponos
        LIBRARY DESTINATION ${INSTALL_PATH}/lib
        ARCHIVE DESTINATION ${INSTALL_PATH}//lib
        )

if(BUILD_SHARED)
	add_library(ponos_shared SHARED ${PONOS_SOURCES} ${PONOS_HEADERS})
	target_include_directories(ponos_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
	target_link_libraries(ponos_shared ${ponos_LIBRARIES} ${PONOS_LIBRARIES})

	install(TARGETS ponos ponos_shared
        LIBRARY DESTINATION ${INSTALL_PATH}/lib
        ARCHIVE DESTINATION ${INSTALL_PATH}//lib
        )
endif(BUILD_SHARED)
