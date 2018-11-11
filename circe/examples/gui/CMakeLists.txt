# set(NANOGUI_INCLUDES 
# "C:/Users/fuiri/Desktop/nanogui/include" 
# "C:/Users/fuiri/Desktop/nanogui/ext/eigen"
# "C:/Users/fuiri/Desktop/nanogui/ext/nanovg/src"
# )
# set(NANOGUI_LIBS "C:/Users/fuiri/Desktop/nanogui/build/nanogui.lib")

add_executable(nanogui_example WIN32 nanogui_example.cpp nanogui_screen.cpp)
target_include_directories(nanogui_example PUBLIC ${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR} ${NANOGUI_INCLUDES} ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(nanogui_example circe ${NANOGUI_LIBS})
target_compile_definitions(nanogui_example PUBLIC GLAD_ALREADY_INCLUDED NANOGUI_GLAD)


if (MSVC)
    set(WINDOWS_BINARIES
        nanogui_example
        )
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
    set_target_properties(${WINDOWS_BINARIES} PROPERTIES
            LINK_FLAGS "/ENTRY:mainCRTStartup")
endif ()