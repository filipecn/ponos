project(PONOS)

set(HEADERS
	src/algorithm/fast_sweep.h
	src/algorithm/marching_squares.h
	src/algorithm/scatter.h
	src/algorithm/search.h
	src/algorithm/sort.h
	src/algorithm/triangulate.h
	src/blas/bicg.h
	src/blas/blas.h
	src/blas/c_regular_grid.h
  src/blas/cg.h
	src/blas/custom_matrix.h
	src/blas/fdm_cg_solver.h
	src/blas/fdm_gauss_seidel_solver.h
	src/blas/fdm_jacobi_solver.h
	src/blas/fdm_matrix.h
	src/blas/fdm_vector.h
	src/blas/field.h
	src/blas/field_grid.h
	src/blas/gauss_jordan.h
	src/blas/linear_system.h
  src/blas/pcg.h
	src/blas/rbf.h
	src/blas/solver_interface.h
	src/blas/sparse_cg_solver.h
	src/blas/sparse_gauss_seidel_solver.h
	src/blas/sparse_jacobi_solver.h
	src/blas/sparse_sor_solver.h
	src/blas/sparse_matrix.h
	src/blas/sparse_vector.h
	src/blas/staggered_grid.h
	src/blas/symmetric_matrix.h
	src/common/defs.h
	src/common/file.h
	src/common/macros.h
	src/common/memory.h
	src/common/random.h
	src/common/timer.h
	src/geometry/bbox.h
	src/geometry/ellipse.h
	src/geometry/frustum.h
	src/geometry/queries.h
	src/geometry/interval.h
	src/geometry/line.h
	src/geometry/matrix.h
	src/geometry/normal.h
	src/geometry/numeric.h
	src/geometry/parametric_surface.h
	src/geometry/point.h
	src/geometry/plane.h
	src/geometry/ray.h
	src/geometry/segment.h
	src/geometry/shape.h
	src/geometry/sphere.h
	src/geometry/surface.h
	src/geometry/quaternion.h
	src/geometry/transform.h
	src/geometry/utils.h
	src/geometry/vector.h
	src/log/debug.h
	src/parallel/parallel.h
	src/spatial/array.h
	src/spatial/spatial_structure_interface.h
	src/structures/bah.h
	src/structures/brep.hpp
	src/structures/grid_interface.h
	src/structures/half_edge.h
	src/structures/level_set.h
	src/structures/mesh.h
	src/structures/regular_grid.h
	src/structures/object_pool.h
	#src/structures/octree.h
	src/structures/quad_tree.h
	src/structures/raw_mesh.h
	src/structures/tetrahedron_mesh.h
	src/structures/z_grid.h
	src/ponos.h
	)

set(SOURCES
	src/algorithm/scatter.cpp
	src/algorithm/triangulate.cpp
	src/common/file.cpp
	src/geometry/bbox.cpp
	src/geometry/queries.cpp
	src/geometry/matrix.cpp
	src/geometry/normal.cpp
	src/geometry/point.cpp
	src/geometry/ray.cpp
	src/geometry/quaternion.cpp
	src/geometry/transform.cpp
	src/geometry/vector.cpp
	src/parallel/parallel.cpp
	src/structures/bah.cpp
	src/structures/level_set.cpp
	src/structures/mesh.cpp
	src/structures/raw_mesh.cpp
	)

include_directories("${PONOS_SOURCE_DIR}/src")
include_directories(${PONOS_INCLUDES})

add_library(ponos ${SOURCES} ${HEADERS})
set_target_properties(ponos PROPERTIES
	OUTPUT_NAME "ponos"
	FOLDER "PONOS")

target_link_libraries(ponos ${ponos_LIBRARIES} ${PONOS_LIBRARIES})

add_subdirectory(examples)

if(TESTPONOS)
  add_subdirectory(tests)
endif(TESTPONOS)

add_subdirectory(tests/unit_tests)
