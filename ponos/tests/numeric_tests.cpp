#include <gtest/gtest.h>

#include <ponos/ponos.h>

using namespace ponos;

TEST(Interpolation, linear) {
  { // 1D
    float dx = 0.01;
    auto f = [](float x) -> float { return cos(x) * sin(x); };
    HaltonSequence sampler;
    for (int i = 0; i < 1000; ++i) {
      auto p = sampler.randomFloat();
      EXPECT_NEAR(lerp<float>(p, f(0), f(dx)), f(p * dx), 1e-6);
    }
  }
  { // 2D
    auto f = [](float x, float y) -> float { return cos(x) * sin(y); };
    RNGSampler sampler;
    float dx = 0.01;
    for (int i = 0; i < 1000; ++i) {
      auto p = sampler.sample(bbox2::unitBox());
      EXPECT_NEAR(bilerp<float>(p.x, p.y,
                                f(0.00, 0.00),
                                f(dx, 0.00),
                                f(dx, dx),
                                f(0.00, dx)),
                  f(p.x * dx, p.y * dx),
                  1e-6);
    }
    { // 3D
      // TODO
    }
  }
}

TEST(Interpolation, monotonicCubic) {
  { // 1D test
    float dx = 0.01;
    auto f = [](float x) -> float { return cos(x) * sin(x); };
    for (float s = 0.0; s <= 1.0; s += 0.01) {
      EXPECT_NEAR(monotonicCubicInterpolate(f(-1 * dx),
                                            f(0),
                                            f(1 * dx),
                                            f(2 * dx),
                                            s),
                  f(s * dx), 1e-7);
    }
  }
  { // 2D test
    float dx = 0.01;
    auto f = [](float x, float y) -> float { return cos(x) * sin(y); };
    float v[4][4];
    for (int s = 0; s < 4; s++)
      for (int u = 0; u < 4; u++)
        v[s][u] = f(s * dx, u * dx);
    RNGSampler sampler;
    for (int i = 0; i < 1000; ++i) {
      auto p = sampler.sample(bbox2::unitBox());
      EXPECT_NEAR(monotonicCubicInterpolate(v, point2(p.x, p.y)),
                  f(dx + p.x * dx, dx + p.y * dx), 1e-7);
    }
  }
  { // 3D test
    float dx = 0.01;
    auto f = [](float x, float y, float z) -> float {
      return cos(x) * sin(y) * sin(z);
    };
    float v[4][4][4];
    for (int s = 0; s < 4; s++)
      for (int u = 0; u < 4; u++)
        for (int w = 0; w < 4; w++)
          v[s][u][w] = f(s * dx, u * dx, w * dx);
    RNGSampler sampler;
    for (int i = 0; i < 1000; ++i) {
      auto p = sampler.sample(bbox3::unitBox());
      EXPECT_NEAR(monotonicCubicInterpolate(v, p),
                  f(dx + p.x * dx, dx + p.y * dx, dx + p.z * dx), 1e-7);
    }
  }
}

TEST(Grid, Methods) {
  { // 2D
    Grid2<float> g(size2(10, 10), vec2(0.1, 0.1), point2(1, 2));
    for (auto e : g.accessor())
      e.value = e.index().i * 10 + e.index().j;
    Grid2<float> gg;
    gg.copy(g);
    EXPECT_EQ(g.resolution(), gg.resolution());
    EXPECT_EQ(g.spacing(), gg.spacing());
    EXPECT_EQ(g.origin(), gg.origin());
    for (auto e : g.accessor())
      EXPECT_NEAR(e.value, gg.accessor()(e.index()), 1e-8);
    g = 7.3f;
    for (auto e : g.accessor())
      EXPECT_NEAR(e.value, 7.3f, 1e-8);
  }
}

TEST(Grid, Iterator) {
  { // 2D
    Grid2<float> g(size2(10, 10));
    g.setSpacing(vec2(0.1, 0.2));
    for (auto e : g.accessor()) {
      EXPECT_NEAR(e.worldPosition().x, e.i() * 0.1, 1e-7);
      EXPECT_NEAR(e.worldPosition().y, e.j() * 0.2, 1e-7);
      EXPECT_NEAR(e.region().size(0), 0.1, 1e-7);
      EXPECT_NEAR(e.region().size(1), 0.2, 1e-7);
      e.value = e.index().i * 10 + e.index().j;
    }
    for (auto e : g.accessor())
      EXPECT_NEAR(e.value, e.index().i * 10 + e.index().j, 1e-8);
  }
}

TEST(Grid, Accessor) {
  { // 2D
    float dx = 0.01;
    Grid2<float> g(size2(10, 10), vec2(dx, dx));
    auto f = [](point2 p) -> float { return std::sin(p.x) * std::cos(p.y); };
    for (auto ij : Index2Range<i32>(g.resolution())) {
      g.accessor()(ij) = f(g.accessor().worldPosition(ij));
      EXPECT_NEAR(g.accessor()(ij), f(g.accessor().worldPosition(ij)), 1e-8);
    }
    // cell info
    EXPECT_NEAR(g.accessor().cellRegion(index2(4, 4)).extends().x, dx, 1e-8);
    EXPECT_NEAR(g.accessor().cellRegion(index2(4, 4)).extends().y, dx, 1e-8);
    EXPECT_EQ(g.accessor().cellIndex(point2(5.5 * dx, 7.8 * dx)), index2(5, 7));
    EXPECT_NEAR(g.accessor().cellPosition(point2(3.1 * dx, 8.7 * dx)).x,
                0.1,
                1e-6);
    EXPECT_NEAR(g.accessor().cellPosition(point2(3.1 * dx, 8.7 * dx)).y,
                0.7,
                1e-6);
    { // CLAMP_TO_EDGE + LINEAR INTERPOLATION
      auto acc =
          g.accessor(AddressMode::CLAMP_TO_EDGE, InterpolationMode::LINEAR);
      EXPECT_NEAR(acc(index2(-5, 5)), f(acc.worldPosition(index2(0, 5))), 1e-8);
      EXPECT_NEAR(acc(index2(5, -5)), f(acc.worldPosition(index2(5, 0))), 1e-8);
      EXPECT_NEAR(acc(index2(11, 5)), f(acc.worldPosition(index2(9, 5))), 1e-8);
      EXPECT_NEAR(acc(index2(5, 11)), f(acc.worldPosition(index2(5, 9))), 1e-8);
      for (index2 ij : Index2Range<i32>(size2(9, 9))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          EXPECT_NEAR(acc(p), f(p), 1e-5);
        }
      }
    }
    { // CLAMP_TO_EDGE + MONOTONIC CUBIC INTERPOLATION
      auto acc =
          g.accessor(AddressMode::CLAMP_TO_EDGE,
                     InterpolationMode::MONOTONIC_CUBIC);
      for (index2 ij : Index2Range<i32>(index2(1, 1), index2(8, 8))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          EXPECT_NEAR(acc(p), f(p), 1e-7);
        }
      }
    }
    {
      auto acc = g.accessor(AddressMode::BORDER, InterpolationMode::LINEAR);
      EXPECT_NEAR(acc(index2(-5, 5)), 0.f, 1e-8);
      EXPECT_NEAR(acc(index2(5, -5)), 0.f, 1e-8);
      EXPECT_NEAR(acc(index2(11, 5)), 0.f, 1e-8);
      EXPECT_NEAR(acc(index2(5, 11)), 0.f, 1e-8);
      for (index2 ij : Index2Range<i32>(size2(9, 9))) {
        RNGSampler sampler;
        for (int i = 0; i < 1; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          EXPECT_NEAR(acc(p), f(p), 1e-5);
        }
      }
    }
    { // BORDER + MONOTONIC CUBIC INTERPOLATION
      auto acc =
          g.accessor(AddressMode::BORDER, InterpolationMode::MONOTONIC_CUBIC);
      for (index2 ij : Index2Range<i32>(index2(1, 1), index2(8, 8))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          EXPECT_NEAR(acc(p), f(p), 1e-7);
        }
      }
    }
  }
}

