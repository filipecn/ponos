link_libraries(aergia)

include_directories("${BOOST_INCLUDE_DIR}")
link_libraries("${BOOST_LIB}" ${AERGIA_LIBRARIES})

set(AAHEADERS
	aaflib/aa.h
	aaflib/aa_rounding.h
	aaflib/aa_interval.h
	aaflib/aa_aaf.h
	aaflib/aa_exceptions.h
	)

set(AASOURCES
	aaflib/aa_aafapprox.cpp
	aaflib/aa_aafspecial.cpp
  aaflib/aa_aafarithm.cpp
	aaflib/aa_aaftrigo.cpp
	aaflib/aa_interval.cpp
	aaflib/aa_aafcommon.cpp
	aaflib/aa_exceptions.cpp
	aaflib/aa_rounding.cpp
	)

include_directories(aaflib)

add_executable(spheres_ia WIN32 spheres_ia.cpp ${AAHEADERS} ${AASOURCES})

if (MSVC)
	# Tell MSVC to use main instead of WinMain for Windows subsystem executables
	set_target_properties(spheres_ia PROPERTIES
		LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()
