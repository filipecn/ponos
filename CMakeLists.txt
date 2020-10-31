# avoid policy warning message
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
# main project
project(PONOS)
cmake_minimum_required(VERSION 3.1)
cmake_policy(VERSION 3.1)
set(CMAKE_VERBOSE_MAKEFILE ON)
# path variables
set(ROOT_PROJECT_PATH "${PROJECT_SOURCE_DIR}")
set(BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR})
##########################################
##               options                ##
##########################################
option(BUILD_ALL "build all libraries" OFF)
option(BUILD_CIRCE "build circe library" OFF)
option(BUILD_HERMES "build hermes library" OFF)
option(BUILD_HELIOS "build helios library" OFF)
option(BUILD_POSEIDON "build poseidon library" OFF)
option(BUILD_ODYSSEUS "build odysseus library" OFF)
option(BUILD_TESTS "build library unit tests" OFF)
option(BUILD_EXAMPLES "build library examples" OFF)
option(BUILD_SHARED "build shared library" OFF)
option(BUILD_DOCS "build library documentation" OFF)
set(INSTALL_PATH ${BUILD_ROOT} CACHE STRING "include and lib folders path")
# cmake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})
# check for specific machine/compiler options.
include(ConfigCompilerFlags)
# include external libraries path
include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ext")
# tests
include(catch2)
# log
include(spdlog)
##########################################
##              projects                ##
##########################################
add_subdirectory(ponos)
#if (BUILD_CIRCE OR BUILD_ALL)
#    include(glfw)
#    include(tinyobj)
#    include(triangle)
#    include(stb)
#    include(glad)
#    include(dirent)
#    add_subdirectory(circe)
#endif (BUILD_CIRCE OR BUILD_ALL)
#if (BUILD_HERMES OR BUILD_ALL)
#    add_subdirectory(hermes)
#endif (BUILD_HERMES OR BUILD_ALL)
#if (BUILD_HELIOS OR BUILD_ALL)
    #   add_subdirectory(helios)
#endif (BUILD_HELIOS OR BUILD_ALL)
#if (BUILD_POSEIDON OR BUILD_ALL)
#    add_subdirectory(poseidon)
#endif (BUILD_POSEIDON OR BUILD_ALL)
##########################################
##         output variables             ##
##########################################
get_directory_property(PONOS_HAS_PARENT PARENT_DIRECTORY)
if (PONOS_HAS_PARENT)
    set(PONOS_INCLUDES ${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR} ${GLAD_INCLUDES} PARENT_SCOPE)
    set(PONOS_EXTRA_LIBS ${GLAD_LIBS} PARENT_SCOPE)
endif (PONOS_HAS_PARENT)
##########################################
##           documentation              ##
##########################################
if(BUILD_DOCS)
    # check if Doxygen is installed
    find_package(Doxygen)
    find_package(Sphinx)
    if (DOXYGEN_FOUND AND SPHINX_FOUND)
        set(DOXYGEN_INPUT_DIRS ${PONOS_SOURCE_DIR}/ponos)
        set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
        set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
        # set input and output files
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        # create doxygen output dir
        file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
        # request to configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message("Doxygen build started")

        # note the option ALL which allows to build the docs together with the application
        add_custom_target( doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM )


        set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/docs/source)
        set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)

        add_custom_target(Sphinx ALL
                      COMMAND ${SPHINX_EXECUTABLE} -b html
                      # Tell Breathe where to find the Doxygen output
                      -Dbreathe_projects.Ponos=${DOXYGEN_OUTPUT_DIR}/xml
                      ${SPHINX_SOURCE} ${SPHINX_BUILD}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      COMMENT "Generating documentation with Sphinx")

    else (DOXYGEN_FOUND AND SPHINX_FOUND)
      message("Doxygen need to be installed to generate the doxygen documentation")
    endif (DOXYGEN_FOUND AND SPHINX_FOUND)
endif(BUILD_DOCS)