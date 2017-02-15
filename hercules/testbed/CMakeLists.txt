link_libraries(aergia)
link_libraries(hercules)

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
