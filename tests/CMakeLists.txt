link_libraries(ponos)

include_directories("${PONOS_SOURCE_DIR}/src")

add_executable(bbox_test bbox_test.cpp)
add_executable(point_test point_test.cpp)
add_executable(ray_test ray_test.cpp)
add_executable(vector_test vector_test.cpp)
