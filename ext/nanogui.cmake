include(ExternalProject)

ExternalProject_Add(
    nanogui
    URL "https://github.com/wjakob/nanogui/archive/master.zip"
    CMAKE_ARGS
            # "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
            # "-DCMAKE_BUILD_TYPE=Release"
            "-DGLFW_BUILD_EXAMPLES=OFF"
            "-DGLFW_BUILD_TESTS=OFF"
            "-DGLFW_BUILD_DOCS=OFF"
            # "-DNANOGUI_INSTALL=OFF"
            "-DNANOGUI_BUILD_EXAMPLE=OFF"
            "-DNANOGUI_BUILD_PYTHON=OFF"
            # "-NANOGUI_BUILD_SHARED"
    CMAKE_CACHE_ARGS
        "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
)

# ExternalProject_Get_Property(nanogui INSTALL_DIR)
set(NANOGUI_INCLUDE_DIR include)
# set(NANOGUI_LIBRARIES
    # ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}nanogui${CMAKE_STATIC_LIBRARY_SUFFIX})

# add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/ext/nanogui" "ext_build/nanogui")
#set_target_properties(nanogui PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
# set_property(TARGET glfw glfw_objects nanogui PROPERTY FOLDER "dependencies")
set(NANOGUI_DEFINITIONS ${NANOGUI_EXTRA_DEFS})
set(NANOGUI_INCLUDES ${NANOGUI_INCLUDE_DIR} ${NANOGUI_EXTRA_INCS})
set(NANOGUI_LIBS ${NANOGUI_LIBRARIES} ${NANOGUI_EXTRA_LIBS})
        

# set(GLFW_INCLUDE_DIR ${GLFW_INCLUDE_DIR} CACHE STRING "")
# set(GLFW_LIBRARIES ${GLFW_LIBRARIES} CACHE STRING "")