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
    // tests against signed distance function defined by phi(x,y) = sqrt(x^2 +
    // y^2) - 1 with the property |grad phi| = 1
    // domain is [-2,2]x[-2,2]
    vec2u r(200);
    vec2f s(4.0 / r.x);
    LevelSet2H ls(r, s, point2f(-2, -2));
    for (auto e : ls.grid().accessor())
      e.value = sqrtf(e.worldPosition().x * e.worldPosition().x +
                      e.worldPosition().y * e.worldPosition().y) -
                1.0;
    auto acc = ls.accessor();
    for (int j = 2; j < r.x - 2; j++)
      for (int i = 2; i < r.x - 2; i++) {
        auto p = acc.worldPosition(i, j);
        float den = sqrtf(p.x * p.x + p.y * p.y);
        if (den == 0)
          continue;
        p /= den;
        float dif[2][2];
        for (int order = 1; order < 4; order++) {
          auto g = acc.gradient(i, j, order);
          dif[order - 1][0] = fabs(g.x - p.x);
          dif[order - 1][1] = fabs(g.y - p.y);
        }
        for (int o = 0; o < 1; o++) {
          EXPECT_LE(dif[o + 1][0], dif[o][0]);
          EXPECT_LE(dif[o + 1][1], dif[o][1]);
        }
        EXPECT_NEAR(acc.gradient(i, j, 3).length(), 1.0f, 1e-2);
      }
  }
}