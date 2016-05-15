include_directories("${PONOS_SOURCE_DIR}/src")

set(HEADERS
core/differential_geometry.h
core/shape.h
geometry/bbox.h
geometry/matrix.h
geometry/normal.h
geometry/point.h
geometry/quaternion.h
geometry/ray.h
geometry/transform.h
geometry/utils.h
geometry/vector.h
log/debug.h
shapes/sphere.h
ponos.h
)

set(SOURCES
core/shape.cpp
geometry/bbox.cpp
geometry/matrix.cpp
geometry/normal.cpp
geometry/point.cpp
geometry/quaternion.cpp
geometry/ray.cpp
geometry/transform.cpp
geometry/vector.cpp
shapes/sphere.cpp
)

add_library(ponos ${SOURCES} ${HEADERS})
set_target_properties(ponos PROPERTIES
                      OUTPUT_NAME "ponos"
                      FOLDER "PONOS")

if (WIN32)
    # The GLFW DLL needs a special compile-time macro and import library name
    #set_target_properties(glfw PROPERTIES PREFIX "" IMPORT_PREFIX "")
    if (MINGW)
        set_target_properties(ponos PROPERTIES IMPORT_SUFFIX "dll.a")
    else()
        set_target_properties(ponos PROPERTIES IMPORT_SUFFIX "dll.lib")
    endif()
endif()

target_link_libraries(ponos ${ponos_LIBRARIES})
