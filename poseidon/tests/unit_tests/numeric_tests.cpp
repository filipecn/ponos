#include <gtest/gtest.h>

#include <poseidon/poseidon.h>

using namespace poseidon;
using namespace ponos;

TEST(VectorGrid, Accessor) {
  { // 2D
    float dx = 0.01;
    VectorGrid2 g(size2(10, 10), vec2(dx, dx));
    for (auto ij : Index2Range<i32>(g.resolution())) {
      g.accessor().u(ij) =
          zalesakDeformationField(g.accessor().uAccessor().worldPosition(ij)).x;
      g.accessor().v(ij) =
          zalesakDeformationField(g.accessor().vAccessor().worldPosition(ij)).y;
      EXPECT_NEAR(g.accessor()(ij).x,
                  zalesakDeformationField(g.accessor().uAccessor().worldPosition(
                      ij)).x,
                  1e-8);
      EXPECT_NEAR(g.accessor()(ij).y,
                  zalesakDeformationField(g.accessor().uAccessor().worldPosition(
                      ij)).y,
                  1e-8);
    }
  }
}

TEST(StaggeredGrid, Accessor) {
  { // 2D
    float dx = 0.01;
    StaggeredGrid2 g(size2(10, 10), vec2(dx, dx));
    EXPECT_NEAR(g.accessor().uAccessor().origin().x, -dx * .5, 1e-8);
    EXPECT_NEAR(g.accessor().uAccessor().origin().y, 0, 1e-8);
    EXPECT_NEAR(g.accessor().vAccessor().origin().x, 0, 1e-8);
    EXPECT_NEAR(g.accessor().vAccessor().origin().y, -dx * .5, 1e-8);
    for (auto ij : Index2Range<i32>(g.resolution())) {
      g.accessor().u(ij) =
          zalesakDeformationField(g.accessor().uAccessor().worldPosition(ij)).x;
      g.accessor().u(ij.plus(1, 0)) =
          zalesakDeformationField(g.accessor().uAccessor().worldPosition(ij.plus(
              1, 0))).x;
      g.accessor().v(ij) =
          zalesakDeformationField(g.accessor().vAccessor().worldPosition(ij)).y;
      g.accessor().v(ij.plus(0, 1)) =
          zalesakDeformationField(g.accessor().vAccessor().worldPosition(ij.plus(
              0, 1))).y;
      EXPECT_NEAR(g.accessor()(ij).x, g.accessor().u(ij), 1e-8);
      EXPECT_NEAR(g.accessor()(ij).y, g.accessor().v(ij), 1e-8);
      EXPECT_NEAR(g.accessor()(ij).x,
                  g.accessor().u(ij) * .5
                      + g.accessor().u(ij.plus(1, 0)) * .5, 1e-8);
      EXPECT_NEAR(g.accessor()(ij).x,
                  (zalesakDeformationField(g.accessor().uAccessor().worldPosition(
                      ij)).x +
                      zalesakDeformationField(g.accessor().uAccessor().worldPosition(
                          ij.plus(1, 0))).x) * .5,
                  1e-8);
      EXPECT_NEAR(g.accessor()(ij).y,
                  g.accessor().v(ij) * .5
                      + g.accessor().v(ij.plus(0, 1)) * .5, 1e-8);
      EXPECT_NEAR(g.accessor()(ij).y,
                  (zalesakDeformationField(g.accessor().vAccessor().worldPosition(
                      ij)).y +
                      zalesakDeformationField(g.accessor().vAccessor().worldPosition(
                          ij.plus(0, 1))).y) * .5,
                  1e-8);
    }
  }
}

