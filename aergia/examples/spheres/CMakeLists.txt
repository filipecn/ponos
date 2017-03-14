link_libraries(aergia)

include_directories("${AERGIA_SOURCE_DIR}/src")

if(WIN32)
	include_directories("C:/boost/include/boost-1_60")
	link_libraries("C:/boost/lib/libboost_system-vc140-mt-gd-1_60.lib" ${GLFW_LIB} ${GLEW_LIB} ${aergia_LIBRARIES} ${OPENGL_LIBRARIES})
else(WIN32)
link_libraries(boost_system ${GLFW_LIB} ${GLEW_LIB} ${aergia_LIBRARIES} ${OPENGL_LIBRARIES})
endif(WIN32)

add_executable(spheres_ia WIN32 spheres_ia.cpp)

if (MSVC)
	# Tell MSVC to use main instead of WinMain for Windows subsystem executables
	set_target_properties(spheres_ia PROPERTIES
		LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()
