project(PONOS)

file(GLOB HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/parallel/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/blas/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/*.h
)

file(GLOB SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithm/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/blas/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/parallel/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/*.cpp
)

include_directories("${PONOS_SOURCE_DIR}/src")
include_directories(${PONOS_INCLUDES})

add_library(ponos ${SOURCES} ${HEADERS})
set_target_properties(ponos PROPERTIES
	OUTPUT_NAME "ponos"
	FOLDER "PONOS")

target_link_libraries(ponos ${ponos_LIBRARIES} ${PONOS_LIBRARIES})

add_subdirectory(examples)
add_subdirectory(tests/unit_tests)
