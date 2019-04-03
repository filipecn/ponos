#ifndef POSEIDON_SIMULATION_CUDA_SMOKE_INJECTOR_H
#define POSEIDON_SIMULATION_CUDA_SMOKE_INJECTOR_H

#include <hermes/hermes.h>
#include <ponos/geometry/point.h>

namespace poseidon {

namespace cuda {

/// Injects different patterns of quantities (ex: smoke concentration,
/// temperature) in grids
class GridSmokeInjector2 {
public:
  /// Injects a circle pattern
  /// \param center circle's center (world coordinates)
  /// \param radius circle's radius (world coordinates)
  /// \param field field reference
  static void injectCircle(const ponos::point2f &center, float radius,
                           hermes::cuda::GridTexture2<float> &field);
};

} // namespace cuda

} // namespace poseidon

#endif // POSEIDON_SIMULATION_CUDA_SMOKE_INJECTOR_H