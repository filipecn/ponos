project(POSEIDON)

enable_language(CUDA)

set(POSEIDON_HEADERS
        poseidon/poseidon.h
        poseidon/math/cuda_fd.h
        poseidon/math/cuda_pcg.h
        poseidon/simulation/cuda_collider.h
        poseidon/simulation/cuda_defs.h
        poseidon/simulation/cuda_integrator.h
        poseidon/simulation/cuda_level_set.h
        poseidon/simulation/cuda_particle_system2.h
        poseidon/simulation/cuda_scene.h
        poseidon/simulation/cuda_smoke_injector.h
        poseidon/simulation/cuda_vector_fields.h
        poseidon/solvers/cuda_practical_liquids_solver2.h
        poseidon/solvers/cuda_practical_liquids_solver3.h
        poseidon/solvers/cuda_smoke_solver.h
        poseidon/solvers/cuda_smoke_solver2_steps.h
        poseidon/solvers/cuda_smoke_solver_kernels.h
        poseidon/solvers/cuda_smoke_solver3_kernels.h
        )
        
set(POSEIDON_SOURCES
        poseidon/math/cuda_fd.cu
        poseidon/math/cuda_pcg.cu
        poseidon/simulation/cuda_eno_integrator2.cu
        poseidon/simulation/cuda_eno_integrator3.cu
        poseidon/simulation/cuda_maccormack_integrator3.cu
        poseidon/simulation/cuda_maccormack_integrator2.cu
        poseidon/simulation/cuda_particle_system2.cu
        poseidon/simulation/cuda_semi_lagrangian_integrator3.cu
        poseidon/simulation/cuda_semi_lagrangian_integrator2.cu
        poseidon/simulation/cuda_smoke_injector.cu
        poseidon/simulation/cuda_vector_fields.cu
        poseidon/solvers/cuda_practical_liquids_solver2.cu
        poseidon/solvers/cuda_practical_liquids_solver3.cu
        poseidon/solvers/cuda_smoke_solver2_steps.cu
        poseidon/solvers/cuda_smoke_solver_kernels.cu
        poseidon/solvers/cuda_smoke_solver3_kernels.cu
        )

add_library(poseidon ${POSEIDON_SOURCES} ${POSEIDON_HEADERS})
set_target_properties(poseidon PROPERTIES
CUDA_STANDARD 11
LINKER_LANGUAGE CUDA
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CMAKE_CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME "poseidon"
        FOLDER "POSEIDON")
target_include_directories(poseidon PUBLIC ${PONOS_SOURCE_DIR} ${HERMES_SOURCE_DIR} ${POSEIDON_SOURCE_DIR})
target_link_libraries(poseidon PUBLIC ponos hermes)


add_subdirectory(examples/cuda)

add_subdirectory(tests/unit_tests)
