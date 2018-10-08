#link_libraries(aergia GLU ${AERGIA_LIBRARIES} ${PONOS_LIBRARIES} ${NANOGUI_LIBS})
add_executable(nanogui_example WIN32 nanogui_example.cpp)
target_include_directories(nanogui_example PUBLIC ${PONOS_SOURCE_DIR} ${AERGIA_SOURCE_DIR} ${GLAD_INCLUDES} ${NANOGUI_INCLUDES} ${EIGEN_INCLUDES} ${NANOVG_INCLUDES})
target_link_libraries(nanogui_example ${NANOGUI_LIBS} aergia ${GLAD_LIBS})


if (MSVC)
    set(WINDOWS_BINARIES
        nanogui_example
        )
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
    set_target_properties(${WINDOWS_BINARIES} PROPERTIES
            LINK_FLAGS "/ENTRY:mainCRTStartup")
endif ()