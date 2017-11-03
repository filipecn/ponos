project(PONOS)

file(GLOB HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/algorithms/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/common/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/parallel/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/blas/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/*.h
)

file(GLOB SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/algorithm/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/geometry/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/blas/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/parallel/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/common/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ponos/structures/*.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PONOS_INCLUDES})

add_library(ponos STATIC ${SOURCES} ${HEADERS})
add_library(ponos_shared SHARED ${SOURCES} ${HEADERS})
set_target_properties(ponos PROPERTIES
	OUTPUT_NAME "ponos"
	FOLDER "PONOS")

target_link_libraries(ponos ${ponos_LIBRARIES} ${PONOS_LIBRARIES})

add_subdirectory(examples)
add_subdirectory(tests/unit_tests)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ponos
	DESTINATION ${ROOT_PROJECT_PATH}/build/include
	FILES_MATCHING REGEX "(.*\\.[inl|h])")
install(TARGETS ponos ponos_shared
	LIBRARY DESTINATION ${ROOT_PROJECT_PATH}/build/lib
	ARCHIVE DESTINATION ${ROOT_PROJECT_PATH}/build/lib
	)
