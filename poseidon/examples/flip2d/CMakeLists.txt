#link_libraries(${GLFW_LIB} ${VULKAN_LIBRARY})

#link_libraries(poseidon)
#include_directories("${POSEIDON_SOURCE_DIR}/src")
#include_directories("${POSEIDON_SOURCE_DIR}/external")
#link_libraries(ponos)
#include_directories("${PONOS_SOURCE_DIR}/src")
#link_libraries(aergia)
#include_directories("${AERGIA_SOURCE_DIR}/src")

link_libraries(${AERGIA_LIBRARIES})
include_directories(${AERGIA_INCLUDES})
add_executable(flip2d hello_flip_2d.cpp flip_2d.h flip_2d.inl flip_2d_model.h)
target_link_libraries(flip2d ${AERGIA_LIBRARIES} -lpthread)

add_executable(view_flip_2d view_flip_2d.cpp)
target_link_libraries(view_flip_2d ${AERGIA_LIBRARIES} -lpthread)


