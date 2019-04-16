#ifndef POSEIDON_SIMULATION_CUDA_SCENE_H
#define POSEIDON_SIMULATION_CUDA_SCENE_H

#include <poseidon/simulation/cuda_collider.h>

namespace poseidon {

namespace cuda {

template <typename T> class Scene2 {
public:
  Collider2<T> **colliders = nullptr;
  Collider2<T> **list = nullptr;
};

} // namespace cuda

} // namespace poseidon

#endif // POSEIDON_SIMULATION_CUDA_SCENE_H