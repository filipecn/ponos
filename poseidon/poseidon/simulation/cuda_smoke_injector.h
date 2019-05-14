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

/// Injects different patterns of quantities (ex: smoke concentration,
/// temperature) in grids
class GridSmokeInjector3 {
public:
  /// Injects a sphere pattern
  /// \param center sphere's center (world coordinates)
  /// \param radius sphere's radius (world coordinates)
  /// \param field field reference
  static void injectSphere(const ponos::point3f &center, float radius,
                           hermes::cuda::RegularGrid3Df &field);
};

} // namespace cuda

} // namespace poseidon

#endif // POSEIDON_SIMULATION_CUDA_SMOKE_INJECTOR_H