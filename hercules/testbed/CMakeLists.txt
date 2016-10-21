link_libraries(aergia)
link_libraries(hercules)

link_libraries(glfw ${aergia_LIBRARIES} ${OPENGL_LIBRARIES})

include_directories("${AERGIA_SOURCE_DIR}/src")
include_directories("${HERCULES_SOURCE_DIR}/src")

include_directories("${HERCULES_SOURCE_DIR}/testbed")

add_executable(testbed main.cpp)

