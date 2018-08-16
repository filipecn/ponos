#link_libraries(aergia GLU ${AERGIA_LIBRARIES} ${PONOS_LIBRARIES} ${NANOGUI_LIBS})
add_executable(nanogui_example WIN32 nanogui_example.cpp)
target_include_directories(nanogui_example PUBLIC ${PONOS_SOURCE_DIR} ${AERGIA_SOURCE_DIR} ${GLAD_INCLUDES} ${NANOGUI_INCLUDES} ${EIGEN_INCLUDES} ${NANOVG_INCLUDES})
target_link_libraries(nanogui_example ${NANOGUI_LIBS} aergia ${GLAD_LIBS})
