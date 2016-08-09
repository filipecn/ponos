project(HELIOS)

set(HEADERS
	src/core/differential_geometry.h
	src/core/intersection.h
	src/core/light.h
	src/core/material.h
	src/core/shape.h
	src/core/primitive.h
	src/geometry/h_ray.h
	src/shapes/h_sphere.h
	src/helios.h
	)

set(SOURCES
	src/core/differential_geometry.cpp
	src/core/shape.cpp
	src/core/primitive.cpp
	src/geometry/h_ray.cpp
	src/shapes/h_sphere.cpp
	)

link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")
include_directories("${HELIOS_SOURCE_DIR}/src")

add_library(helios ${SOURCES} ${HEADERS})
set_target_properties(helios PROPERTIES
	OUTPUT_NAME "helios"
	FOLDER "HELIOS")

target_link_libraries(helios ${helios_LIBRARIES})
