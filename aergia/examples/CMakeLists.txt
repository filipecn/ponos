link_libraries(aergia)

link_libraries(glfw ${aergia_LIBRARIES} ${OPENGL_LIBRARIES})

include_directories("${AERGIA_SOURCE_DIR}/src")

add_executable(hello_aergia_test WIN32 hello_aergia.cpp)

add_executable(hello_3d WIN32 hello3D.cpp)
add_executable(hello_app WIN32 hello_app.cpp)
set(WINDOWS_BINARIES hello_aergia_test)

if (MSVC)
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
    set_target_properties(${WINDOWS_BINARIES} PROPERTIES
                          LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()


add_executable(openvdbexample openvdbexample.cpp)
include_directories("/home/filipecn/gitclones/OpenVDB/include")
target_link_libraries(openvdbexample ${aergia_LIBRARIES} -ltbb
	/home/filipecn/gitclones/OpenVDB/lib/libopenvdb.so
	/home/filipecn/gitclones/OpenVDB/lib/libopenvdb_points.so)
