project(POSEIDON)

enable_language(CUDA)

set(POSEIDON_HEADERS
        poseidon/poseidon.h
        poseidon/simulation/cuda_collider.h
        poseidon/simulation/cuda_scene.h
        poseidon/simulation/cuda_smoke_injector.h
        poseidon/solvers/cuda_smoke_solver.h
        )
        
set(POSEIDON_SOURCES
        poseidon/simulation/cuda_smoke_injector.cu
        poseidon/solvers/cuda_smoke_solver.cu
        )

add_library(poseidon ${POSEIDON_SOURCES} ${POSEIDON_HEADERS})
set_target_properties(poseidon PROPERTIES
CUDA_STANDARD 11
LINKER_LANGUAGE CUDA
CMAKE_CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME "poseidon"
        FOLDER "POSEIDON")
target_include_directories(poseidon PUBLIC ${PONOS_SOURCE_DIR} ${HERMES_SOURCE_DIR} ${POSEIDON_SOURCE_DIR})
target_link_libraries(poseidon PUBLIC ponos hermes)

add_subdirectory(examples/cuda/smoke2d)

    #	add_subdirectory(tests)
