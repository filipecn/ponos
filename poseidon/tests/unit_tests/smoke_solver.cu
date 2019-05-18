#include <gtest/gtest.h>
#include <poseidon/poseidon.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;
using namespace poseidon::cuda;

TEST(SmokeSolver3, PressureSystem) {
  // y
  // |
  //  ---x
  // z = 0
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  // z = 1
  //  S S S S  - - - -
  //  S F F S  - 2 3 -
  //  S F F S  - 0 1 -
  //  S S S S  - - - -
  // z = 2
  //  S S S S  - - - -
  //  S F F S  - 6 7 -
  //  S F F S  - 4 5 -
  //  S S S S  - - - -
  // z = 3
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  vec3u size(4);
  RegularGrid3Huc h_solid(size);
  RegularGrid3Hf h_div(size);
  auto hsAcc = h_solid.accessor();
  auto dAcc = h_div.accessor();
  int divIndex = 0;
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++) {
        dAcc(i, j, k) = 0;
        if (i > 0 && i < size.x - 1 && j > 0 && j < size.y - 1 && k > 0 &&
            k < size.z - 1) {
          dAcc(i, j, k) = divIndex++;
          hsAcc(i, j, k) = 0;
        } else
          hsAcc(i, j, k) = 1;
      }
  RegularGrid3Duc d_solid(size);
  memcpy(d_solid.data(), h_solid.data());
  RegularGrid3Df d_div(size);
  d_div.setSpacing(vec3f(0.01f));
  memcpy(d_div.data(), h_div.data());
  FDMatrix3D d_A(size);
  MemoryBlock1Df d_rhs;
  size_t systemSize = setupPressureSystem(d_div, d_solid, d_A, 0.001, d_rhs);
  MemoryBlock1Hf h_rhs(d_rhs.size());
  h_rhs.allocate();
  memcpy(h_rhs, d_rhs);
  std::cerr << h_rhs << std::endl;
  FDMatrix3H h_A(size);
  h_A.copy(d_A);
  std::cerr << h_A << std::endl;
}
