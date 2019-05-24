#ifndef POSEIDON_SIMULATION_CUDA_SCENE_H
#define POSEIDON_SIMULATION_CUDA_SCENE_H

#include <hermes/numeric/cuda_grid.h>
#include <poseidon/simulation/cuda_collider.h>

namespace poseidon {

namespace cuda {

template <typename T> class Scene2 {
public:
  Collider2<T> **colliders = nullptr;
  Collider2<T> **list = nullptr;
};

template <typename T> class Scene3 {
public:
  Collider3<T> **colliders = nullptr;
  Collider3<T> **list = nullptr;
  void resize(hermes::cuda::vec3u size) {
    target_temperature.resize(size);
    smoke_source.resize(size);
  }
  void setSpacing(hermes::cuda::vec3f spacing) {
    target_temperature.setSpacing(spacing);
    smoke_source.setSpacing(spacing);
  }
  hermes::cuda::RegularGrid3Df target_temperature;
  hermes::cuda::RegularGrid3Duc smoke_source;
};

} // namespace cuda

} // namespace poseidon

#endif // POSEIDON_SIMULATION_CUDA_SCENE_H