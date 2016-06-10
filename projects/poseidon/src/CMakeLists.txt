include_directories("${POSEIDON_SOURCE_DIR}/src")
include_directories("${POSEIDON_SOURCE_DIR}/external")

link_libraries(ponos)

include_directories("${PONOS_SOURCE_DIR}/src")

include_directories("C:/Users/fuiri/Downloads")

set(HEADERS
algorithms/flip.h
elements/particle.h
math/conjugate_gradient.h
structures/particle_grid.h
structures/particle_set.h
poseidon.h
)

set(SOURCES
algorithms/flip.cpp
elements/particle.cpp
math/conjugate_gradient.cpp
structures/particle_grid.cpp
)

add_library(poseidon ${SOURCES} ${HEADERS})
set_target_properties(poseidon PROPERTIES
                      OUTPUT_NAME "poseidon"
                    FOLDER "POSEIDON")

if (WIN32)
    # The GLFW DLL needs a special compile-time macro and import library name
    #set_target_properties(glfw PROPERTIES PREFIX "" IMPORT_PREFIX "")
    if (MINGW)
        set_target_properties(poseidon PROPERTIES IMPORT_SUFFIX "dll.a")
    else()
        set_target_properties(poseidon PROPERTIES IMPORT_SUFFIX "dll.lib")
    endif()
endif()

target_link_libraries(poseidon ${poseidon_LIBRARIES})
