#include <gtest/gtest.h>
#include <poseidon/poseidon.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;
using namespace poseidon::cuda;

TEST(LevelSet2, gradient) {
  {
    float res = 0.01;
    LevelSet2H ls(vec2u(10), vec2f(res));
    for (auto e : ls.grid().accessor())
      e.value = e.worldPosition().x;
    auto acc = ls.accessor();
    for (int j = 1; j < 9; j++)
      for (int i = 1; i < 9; i++) {
        auto p = acc.worldPosition(i, j);
        EXPECT_NEAR(acc.gradient(i, j).x, 1.f, res * res);
        EXPECT_NEAR(acc.gradient(i, j).y, 0.f, res * res);
      }
  }
  {
    float res = 0.01;
    LevelSet2H ls(vec2u(10), vec2f(res));
    for (auto e : ls.grid().accessor())
      e.value = e.worldPosition().y;
    auto acc = ls.accessor();
    for (int j = 2; j < 8; j++)
      for (int i = 2; i < 8; i++) {
        auto p = acc.worldPosition(i, j);
        EXPECT_NEAR(acc.gradient(i, j).x, 0.f, res * res);
        EXPECT_NEAR(acc.gradient(i, j).y, 1.f, res * res);
      }
  }
  {
    float res = 0.01;
    LevelSet2H ls(vec2u(10), vec2f(res));
    float center = 5 * res;
    float radius = 1 * res;
    for (auto e : ls.grid().accessor())
      e.value = (e.worldPosition() - point2f(center)).length() - radius;
    auto acc = ls.accessor();
    std::cerr << ls.grid().data() << std::endl;
    for (int j = 2; j < 100; j++)
      for (int i = 2; i < 100; i++) {
        auto p = acc.worldPosition(i, j);
        EXPECT_NEAR(acc.gradient(i, j).x, 2 * (p.x - center), res * res);
        EXPECT_NEAR(acc.gradient(i, j).y, 2 * (p.y - center), res * res);
        return;
      }
  }
}