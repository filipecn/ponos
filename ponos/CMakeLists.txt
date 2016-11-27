project(PONOS)

set(HEADERS
	src/algorithm/search.h
	src/algorithm/sort.h
	src/common/defs.h
	src/common/file.h
	src/common/macros.h
	src/common/memory.h
	src/common/random.h
	src/common/timer.h
	src/geometry/bbox.h
	src/geometry/frustum.h
	src/geometry/queries.h
	src/geometry/interval.h
	src/geometry/line.h
	src/geometry/matrix.h
	src/geometry/normal.h
	src/geometry/numeric.h
	src/geometry/point.h
	src/geometry/plane.h
	src/geometry/ray.h
	src/geometry/segment.h
	src/geometry/shape.h
	src/geometry/sphere.h
	src/geometry/quaternion.h
	src/geometry/transform.h
	src/geometry/utils.h
	src/geometry/vector.h
	src/log/debug.h
	src/parallel/parallel.h
	src/spatial/array.h
	src/spatial/spatial_structure_interface.h
	src/structures/brep.hpp
	src/structures/c_grid_interface.h
	src/structures/c_regular_grid.h
	src/structures/regular_grid.h
	src/structures/object_pool.h
	src/structures/z_grid.h
	src/ponos.h
	)

set(SOURCES
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
	)

include_directories("${PONOS_SOURCE_DIR}/src")

add_library(ponos ${SOURCES} ${HEADERS})
set_target_properties(ponos PROPERTIES
	OUTPUT_NAME "ponos"
	FOLDER "PONOS")

target_link_libraries(ponos ${ponos_LIBRARIES})

if(TESTPONOS)
  add_subdirectory(tests)
endif(TESTPONOS)
