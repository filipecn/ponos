#include <gtest/gtest.h>
#include <poseidon/poseidon.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;
using namespace poseidon::cuda;

TEST(ParticleSystem2, acc) {
  ParticleSystem2H h_ps;
  h_ps.resize(1000);
  size_t i = 0;
  for (auto p : h_ps.accessor()) {
    EXPECT_EQ(i++, p.id());
    EXPECT_TRUE(p.isActive());
    p.position = point2f(p.id() / 1000.f);
  }
  ParticleSystem2D d_ps(h_ps);
}
