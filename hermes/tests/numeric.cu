#include "utils.h"
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;

TEST(Interpolation, monotonicCubic) {
  { // 1D test
    auto f = [](float x) -> float { return cos(x) * sin(x); };
    for (float s = 0.0; s <= 1.0; s += 0.01) {
      EXPECT_NEAR(monotonicCubicInterpolate(f(-0.1), f(0.0), f(0.1), f(0.2), s),
                  f(s * 0.1), 0.1 * 0.1);
    }
    for (float s = 0.0; s <= 1.0; s += 0.01)
      EXPECT_NEAR(
          monotonicCubicInterpolate(f(-0.01), f(0.0), f(0.01), f(0.02), s),
          f(s * 0.01), 0.01 * 0.01);
  }
  vec3u size(16);
  RegularGrid3Hf h(size);
  h.setSpacing(vec3f(0.01));
  auto hAcc = h.accessor();
  for (size_t k = 0; k < size.z; k++)
    for (size_t j = 0; j < size.y; j++)
      for (size_t i = 0; i < size.x; i++) {
        auto p = hAcc.worldPosition(i, j, k);
        // hAcc(i, j, k) = cosf(p.x) * cosf(p.y) * cosf(p.z);
        hAcc(i, j, k) = p.x;
      }
  for (size_t k = 1; k < size.z - 3; k++)
    for (size_t j = 1; j < size.y - 3; j++)
      for (size_t i = 1; i < size.x - 3; i++) {
        EXPECT_NEAR(hAcc(hAcc.worldPosition(i, j, k)), hAcc(i, j, k), 1e-8);
      }
  HaltonSequence sx(3), sy(5), sz(7);
  for (size_t k = 1; k < size.z - 3; k++)
    for (size_t j = 1; j < size.y - 3; j++)
      for (size_t i = 1; i < size.x - 3; i++) {
        point3f p(0.01 * i + sx.randomFloat() * 0.01,
                  0.01 * j + sy.randomFloat() * 0.01,
                  0.01 * k + sz.randomFloat() * 0.01);
        // EXPECT_NEAR(cosf(samples[i].x) * cosf(samples[i].y) *
        // cosf(samples[i].z),
        EXPECT_NEAR(p.x, hAcc(p), 0.01 * 0.01);
      }
}