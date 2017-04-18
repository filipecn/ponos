link_libraries(aergia)
link_libraries(hercules)

set(FREETYPE_LIBRARIES ${FREETYPE_LIB} z)
set(FREETYPE_INCLUDE "/usr/local/include/freetype2")
include_directories("${FREETYPE_INCLUDE}")
link_libraries(${FREETYPE_LIBRARIES})
include_directories("C:/VulkanSDK/1.0.3.1/Include")
include_directories("${AERGIA_SOURCE_DIR}/external/GLEW")
include_directories("${AERGIA_SOURCE_DIR}/external/freeglut")
include_directories("${AERGIA_SOURCE_DIR}/external")
set(OPENGL_LIBRARIES "${AERGIA_SOURCE_DIR}/external/freeglut/x64/freeglut.lib")

link_libraries(${GLFW_LIB} ${aergia_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIB})

include_directories("${AERGIA_SOURCE_DIR}/src")
include_directories("${HERCULES_SOURCE_DIR}/src")

include_directories("${HERCULES_SOURCE_DIR}/testbed")

add_executable(testbed main.cpp)
