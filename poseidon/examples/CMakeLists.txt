if(WIN32)
  set(CMAKE_MODULE_PATH "${AERGIA_SOURCE_DIR}/CMake/modules")
  find_library(GLFW_LIB NAMES glfw3 PATHS ${AERGIA_SOURCE_DIR}/external/GLFW)
	include_directories("C:/VulkanSDK/1.0.3.1/Include")
	include_directories("${AERGIA_SOURCE_DIR}/external")
	include_directories("${AERGIA_SOURCE_DIR}/external/freeglut")
	include_directories("${AERGIA_SOURCE_DIR}/external/GLEW")
  set(GLEW_LIB "${AERGIA_SOURCE_DIR}/external/GLEW/x64/glew32.lib")
  set(OPENGL_LIBRARIES "${AERGIA_SOURCE_DIR}/external/freeglut/x64/freeglut.lib")
	find_library(VULKAN_LIB NAMES vulkan-1 vulkan)
	find_package(Vulkan REQUIRED)
	find_package(OpenGL REQUIRED)
	include_directories(${OPENGL_INCLUDE_DIRS})
	link_libraries(${GLFW_LIB} ${VULKAN_LIBRARY} opengl32 glu32)
  include_directories("C:/Users/fuiri/Downloads")
endif(WIN32)

find_library(LAPACK_LIB NAMES liblapack PATHS ${POSEIDON_SOURCE_DIR}/external)

link_libraries(${GLFW_LIB} ${GLEW_LIB} ${aergia_LIBRARIES} ${OPENGL_LIBRARIES})
link_libraries(${GLFW_LIB} ${aergia_LIBRARIES} ${VULKAN_LIBRARY} ${POSEIDON_SOURCE_DIR}/external/libcblas.dll.a)

link_libraries(poseidon)
include_directories("${POSEIDON_SOURCE_DIR}/src")
include_directories("${POSEIDON_SOURCE_DIR}/external")
link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")
link_libraries(aergia)
include_directories("${AERGIA_SOURCE_DIR}/src")

include_directories("${POSEIDON_SOURCE_DIR}/examples")

add_executable(zgrid WIN32 zgrid.cpp)

set(WINDOWS_BINARIES zgrid)

if (MSVC)
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
		set_target_properties(${WINDOWS_BINARIES} PROPERTIES
		                     LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()

add_subdirectory(flip2d)

if(NOT WIN32)
add_subdirectory(openvdb)
endif(NOT WIN32)
