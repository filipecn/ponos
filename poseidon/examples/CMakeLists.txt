if (WIN32)
  find_library(GLFW_LIB NAMES glfw3 PATHS ${AERGIA_SOURCE_DIR}/external/GLFW)
  include_directories("${AERGIA_SOURCE_DIR}/external")
  #find_library(VULKAN_LIB NAMES vulkan-1 vulkan)
  find_package(Vulkan REQUIRED)
  #message(FATAL_ERROR "${GLFW_LIB}")
endif()

find_library(LAPACK_LIB NAMES liblapack PATHS ${POSEIDON_SOURCE_DIR}/external)

link_libraries(${GLFW_LIB} ${aergia_LIBRARIES} ${VULKAN_LIBRARY} ${POSEIDON_SOURCE_DIR}/external/libcblas.dll.a)

# link_libraries(${GLFW_LIB} ${VULKAN_LIBRARY} opengl32 glu32)

link_libraries(poseidon)
include_directories("${POSEIDON_SOURCE_DIR}/src")
include_directories("${POSEIDON_SOURCE_DIR}/external")
link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")
link_libraries(aergia)
include_directories("${AERGIA_SOURCE_DIR}/src")

include_directories("C:/Users/fuiri/Downloads")

include_directories("${POSEIDON_SOURCE_DIR}/examples")

# add_executable(hello_flip_test WIN32 hello_flip.cpp flip_drawer.h)

set(WINDOWS_BINARIES hello_flip_test)

if (MSVC)
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
    set_target_properties(${WINDOWS_BINARIES} PROPERTIES
                          LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()

add_subdirectory(openvdb)
