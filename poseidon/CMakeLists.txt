project(POSEIDON)

set(POSEIDON_HEADERS
        poseidon/poseidon.h
        poseidon/cuda/numeric/cuda_fd.h
        poseidon/cuda/numeric/cuda_pcg.h
        poseidon/cuda/simulation/cuda_collider.h
        poseidon/cuda/simulation/cuda_defs.h
        poseidon/cuda/simulation/cuda_integrator.h
        poseidon/cuda/simulation/cuda_level_set.h
        poseidon/cuda/simulation/cuda_particle_system2.h
        poseidon/cuda/simulation/cuda_scene.h
        poseidon/cuda/simulation/cuda_smoke_injector.h
        poseidon/cuda/numeric/cuda_vector_fields.h
        poseidon/cuda/solvers/cuda_practical_liquids_solver2.h
        poseidon/cuda/solvers/cuda_practical_liquids_solver3.h
        poseidon/cuda/solvers/cuda_smoke_solver.h
        poseidon/cuda/solvers/cuda_smoke_solver2_steps.h
        poseidon/cuda/solvers/cuda_smoke_solver_kernels.h
        poseidon/cuda/solvers/cuda_smoke_solver3_kernels.h

        poseidon/numeric/staggered_grid.h
        poseidon/numeric/vector_fields.h
        poseidon/numeric/vector_grid.h
        poseidon/simulation/integrators/integrator_interface.h
        poseidon/simulation/integrators/semi_lagrangian_integrator.h
        poseidon/solvers/grid_smoke_solver.h
        )

set(POSEIDON_CUDA_SOURCES
        poseidon/cuda/numeric/cuda_fd.cu
        poseidon/cuda/numeric/cuda_pcg.cu
        poseidon/cuda/simulation/cuda_eno_integrator2.cu
        poseidon/cuda/simulation/cuda_eno_integrator3.cu
        poseidon/cuda/simulation/cuda_maccormack_integrator3.cu
        poseidon/cuda/simulation/cuda_maccormack_integrator2.cu
        poseidon/cuda/simulation/cuda_particle_system2.cu
        poseidon/cuda/simulation/cuda_semi_lagrangian_integrator3.cu
        poseidon/cuda/simulation/cuda_semi_lagrangian_integrator2.cu
        poseidon/cuda/simulation/cuda_smoke_injector.cu
        poseidon/cuda/numeric/cuda_vector_fields.cu
        poseidon/cuda/solvers/cuda_practical_liquids_solver2.cu
        poseidon/cuda/solvers/cuda_practical_liquids_solver3.cu
        poseidon/cuda/solvers/cuda_smoke_solver2_steps.cu
        poseidon/cuda/solvers/cuda_smoke_solver_kernels.cu
        poseidon/cuda/solvers/cuda_smoke_solver3_kernels.cu
        )

set(POSEIDON_SOURCES
        poseidon/numeric/staggered_grid.cpp
        poseidon/numeric/vector_fields.cpp
        poseidon/numeric/vector_grid.cpp
        poseidon/simulation/integrators/semi_lagrangian_integrator.cpp
        poseidon/solvers/grid_smoke_solver.cpp
        )

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

enable_language("CUDA")
list(APPEND CUDA_NVCC_FLAGS -arch compute_30 -use_fast_math)
add_definitions(-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1)
if (MSVC)
    list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif (MSVC)

add_library(poseidon ${POSEIDON_SOURCES} ${POSEIDON_HEADERS} ${POSEIDON_CUDA_SOURCES})
set_target_properties(poseidon PROPERTIES
        CUDA_STANDARD 11
        LINKER_LANGUAGE CUDA
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CMAKE_CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME "poseidon"
        FOLDER "POSEIDON")
target_include_directories(poseidon PUBLIC
        ${PONOS_SOURCE_DIR}
        ${HERMES_SOURCE_DIR}
        ${POSEIDON_SOURCE_DIR})
target_link_libraries(poseidon PUBLIC ponos hermes)

if (BUILD_EXAMPLES)
    add_subdirectory(examples/shallow_waters)
    add_subdirectory(examples/smoke_sim)
endif (BUILD_EXAMPLES)

if (BUILD_TESTS)
    add_subdirectory(tests/cuda_unit_tests)
    add_subdirectory(tests/unit_tests)
endif (BUILD_TESTS)


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/poseidon
        DESTINATION ${INSTALL_PATH}/include
        FILES_MATCHING REGEX "(.*\\.[inl|h])")
install(DIRECTORY poseidon/cuda
        DESTINATION ${INSTALL_PATH}/include
        FILES_MATCHING REGEX "(.*\\.[inl|h])")

install(TARGETS poseidon
        LIBRARY DESTINATION ${INSTALL_PATH}/lib
        ARCHIVE DESTINATION ${INSTALL_PATH}//lib
        )
