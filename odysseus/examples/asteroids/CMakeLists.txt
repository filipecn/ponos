link_libraries(odysseus)

link_libraries(glfw ${aergia_LIBRARIES} ${OPENGL_LIBRARIES})

include_directories("${ODYSSEUS_SOURCE_DIR}/src")

include_directories("${AERGIA_SOURCE_DIR}/src")

add_executable(asteroids asteroids.cpp
	player.h
	bullet_pool.hpp
	game.hpp
	)

