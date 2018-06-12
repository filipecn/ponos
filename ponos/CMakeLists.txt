project(PONOS)

set(PONOS_PONOS_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/ponos.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/algorithm/fast_sweep.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/algorithm/geometric_shapes.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/algorithm/marching_squares.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/algorithm/scatter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/algorithm/sort.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/algorithm/triangulate.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/bicg.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/blas.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/cg.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/c_regular_grid.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/c_regular_grid.inl
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/custom_matrix.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/fdm_cg_solver.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/fdm_gauss_seidel_solver.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/fdm_jacobi_solver.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/fdm_matrix.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/fdm_vector.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/field_grid.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/field_grid.inl
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/field.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/gauss_jordan.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/interpolator.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/interpolator.inl
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/kernels.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/linear_system.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/pcg.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/rbf.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/solver_interface.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/sparse_cg_solver.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/sparse_gauss_seidel_solver.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/sparse_jacobi_solver.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/sparse_matrix.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/sparse_sor_solver.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/sparse_vector.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/staggered_grid.h
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/staggered_grid.inl
        ${CMAKE_CURRENT_SOURCE_DIR}ponos/blas/symmetric_matrix.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/common/defs.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/common/file.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/common/macros.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/common/memory.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/common/parallel.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/common/random.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/common/timer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/bbox.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/ellipse.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/frustum.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/interval.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/line.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/matrix.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/normal.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/numeric.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/parametric_surface.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/plane.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/point.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/polygon.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/quaternion.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/queries.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/ray.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/segment.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/shape.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/sphere.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/surface.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/transform.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/utils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/vector.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/log/debug.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/parallel/parallel.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/bah.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/brep.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/grid_interface.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/half_edge3.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/half_edge.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/level_set.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/mesh.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/object_pool.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/octree.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/octree.inl
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/point_set_interface.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/quad_tree.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/quad_tree.inl
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/raw_mesh.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/regular_grid.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/tetrahedron_mesh.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/tetrahedron_mesh.inl
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/z_grid.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/z_point_set.h
        )

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/algorithm/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/blas/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/parallel/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/common/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/*.cpp
        )

add_library(ponos STATIC ${SOURCES} ${HEADERS})
set_target_properties(ponos PROPERTIES
        OUTPUT_NAME "ponos"
        FOLDER "PONOS")
target_include_directories(ponos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PONOS_INCLUDES})
target_link_libraries(ponos ${ponos_LIBRARIES} ${PONOS_LIBRARIES})

add_library(ponos_shared SHARED ${SOURCES} ${HEADERS})
target_include_directories(ponos_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PONOS_INCLUDES})
target_link_libraries(ponos_shared ${ponos_LIBRARIES} ${PONOS_LIBRARIES})

# add_subdirectory(examples)
add_subdirectory(tests)

if (WIN32)
    SET(INSTALL_PATH ${ROOT_PROJECT_PATH}/wbuild)
else (WIN32)
    SET(INSTALL_PATH ${ROOT_PROJECT_PATH}/build)
endif (WIN32)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ponos
        DESTINATION ${INSTALL_PATH}/include
        FILES_MATCHING REGEX "(.*\\.[inl|h])")

install(TARGETS ponos ponos_shared
        LIBRARY DESTINATION ${INSTALL_PATH}/lib
        ARCHIVE DESTINATION ${INSTALL_PATH}//lib
        )
