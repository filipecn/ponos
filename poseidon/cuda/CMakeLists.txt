project(POSEIDON)

enable_language(CUDA)

set(POSEIDON_HEADERS
        poseidon/poseidon.h
        cuda/numeric/cuda_fd.h
        cuda/numeric/cuda_pcg.h
        cuda/simulation/cuda_collider.h
        cuda/simulation/cuda_defs.h
        cuda/simulation/cuda_integrator.h
        cuda/simulation/cuda_level_set.h
        cuda/simulation/cuda_particle_system2.h
        cuda/simulation/cuda_scene.h
        cuda/simulation/cuda_smoke_injector.h
        cuda/numeric/cuda_vector_fields.h
        cuda/solvers/cuda_practical_liquids_solver2.h
        cuda/solvers/cuda_practical_liquids_solver3.h
        cuda/solvers/cuda_smoke_solver.h
        cuda/solvers/cuda_smoke_solver2_steps.h
        cuda/solvers/cuda_smoke_solver_kernels.h
        cuda/solvers/cuda_smoke_solver3_kernels.h
        )

set(POSEIDON_SOURCES
        cuda/numeric/cuda_fd.cu
        cuda/numeric/cuda_pcg.cu
        cuda/simulation/cuda_eno_integrator2.cu
        cuda/simulation/cuda_eno_integrator3.cu
        cuda/simulation/cuda_maccormack_integrator3.cu
        cuda/simulation/cuda_maccormack_integrator2.cu
        cuda/simulation/cuda_particle_system2.cu
        cuda/simulation/cuda_semi_lagrangian_integrator3.cu
        cuda/simulation/cuda_semi_lagrangian_integrator2.cu
        cuda/simulation/cuda_smoke_injector.cu
        cuda/numeric/cuda_vector_fields.cu
        cuda/solvers/cuda_practical_liquids_solver2.cu
        cuda/solvers/cuda_practical_liquids_solver3.cu
        cuda/solvers/cuda_smoke_solver2_steps.cu
        cuda/solvers/cuda_smoke_solver_kernels.cu
        cuda/solvers/cuda_smoke_solver3_kernels.cu
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

if (BUILD_EXAMPLES)
    # add_subdirectory(examples/cuda)
    add_subdirectory(examples/shallow_waters)
endif (BUILD_EXAMPLES)
if (BUILD_TESTS)
    add_subdirectory(tests/unit_tests)
endif (BUILD_TESTS)

