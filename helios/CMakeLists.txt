project(HELIOS)

set(HEADERS
		helios/accelerators/aggregate.h
		helios/accelerators/h_bvh.h
		#helios/accelerators/grid.h
		#helios/cameras/orthographic.h
		#helios/cameras/perspective.h
		helios/common/e_float.h
		helios/common/utils.h
		#helios/core/bsdf.h
		#helios/core/camera.h
		#helios/core/differential_geometry.h
		#helios/core/film.h
		helios/core/interaction.h
		#helios/core/intersection.h
		#helios/core/light.h
		#helios/core/material.h
		#helios/core/sampler.h
		helios/core/shape.h
		helios/core/spectrum.h
		helios/core/spectrum.inl
		helios/core/primitive.h
		#helios/core/texture.h
		helios/geometry/bounds.h
		helios/geometry/bounds.inl
		helios/geometry/queries.h
		helios/geometry/h_ray.h
		helios/shapes/h_sphere.h
		helios/shapes/triangle_mesh.h
		helios/helios.h
	)

set(SOURCES
		helios/accelerators/h_bvh.cpp
		#helios/accelerators/grid.cpp
		#helios/cameras/orthographic.cpp
		#helios/cameras/perspective.cpp
		#helios/core/bsdf.cpp
		#helios/core/camera.cpp
		#helios/core/intersection.cpp
		helios/core/shape.cpp
		helios/core/spectrum.cpp
		helios/core/primitive.cpp
		helios/geometry/queries.cpp
		helios/geometry/h_ray.cpp
		helios/shapes/h_sphere.cpp
		helios/shapes/triangle_mesh.cpp
        helios/core/interaction.cpp
		helios/common/e_float.cpp helios/geometry/h_transform.h 
		helios/geometry/h_transform.cpp helios/common/globals.h 
		helios/common/globals.cpp)

add_library(helios STATIC ${SOURCES} ${HEADERS})
set_target_properties(helios PROPERTIES
		OUTPUT_NAME "helios"
		FOLDER "HELIOS")
target_link_libraries(helios PUBLIC ponos)
target_include_directories(helios PUBLIC ${PONOS_SOURCE_DIR} ${HELIOS_SOURCE_DIR})
