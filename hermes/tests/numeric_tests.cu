#include <catch2/catch.hpp>

#include <hermes/hermes.h>

using namespace hermes::cuda;

TEST_CASE("Interpolation", "[numeric][interpolation]") {
  SECTION("monotonicCubic") {
    { // 1D test
      auto f = [](float x) -> float { return cos(x) * sin(x); };
      for (float s = 0.0; s <= 1.0; s += 0.01) {
        REQUIRE(monotonicCubicInterpolate(f(-0.1), f(0.0), f(0.1), f(0.2), s) ==
                Approx(f(s * 0.1)).margin(0.1 * 0.1));
      }
      for (float s = 0.0; s <= 1.0; s += 0.01)
        REQUIRE(
            monotonicCubicInterpolate(f(-0.01), f(0.0), f(0.01), f(0.02), s) ==
            Approx(f(s * 0.01)).margin(0.01 * 0.01));
    }
    {
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
            REQUIRE(hAcc(hAcc.worldPosition(i, j, k)) ==
                    Approx(hAcc(i, j, k)).margin(1e-8));
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
            REQUIRE(p.x == Approx(hAcc(p)).margin(0.01 * 0.01));
          }
    }
    {
      vec2u size(16);
      RegularGrid2Hf h(size);
      h.setSpacing(vec2f(0.01));
      auto hAcc = h.accessor();
      for (auto e : hAcc) {
        auto p = e.worldPosition();
        e.value = cosf(p.x) * cosf(p.y);
        // e.value = p.x;
      }
      for (size_t j = 1; j < size.y - 2; j++)
        for (size_t i = 1; i < size.x - 2; i++) {
          REQUIRE(hAcc(hAcc.worldPosition(i, j)) ==
                  Approx(hAcc(i, j)).margin(1e-8));
        }
      HaltonSequence sx(2), sy(5);
      for (size_t j = 1; j < size.y - 2; j++)
        for (size_t i = 1; i < size.x - 2; i++) {
          point2f p(0.01 * i + sx.randomFloat() * 0.01,
                    0.01 * j + sy.randomFloat() * 0.01);
          REQUIRE(cosf(p.x) * cosf(p.y) == Approx(hAcc(p)).margin(0.01 * 0.01));
        }
    }
  }
}

struct map_ipj {
  __device__ void operator()(index2 index, float &value) const {
    value = index.i * 10 + index.j;
  }
};

TEST_CASE("Grid-sanity", "[numeric][grid][access]") {
  SECTION("2d") {
    Grid2<float> g(size2(10, 10), vec2(0.1, 0.1), point2(1, 2));
    g = 3.0;
    ponos::Grid2<float> hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(3).margin(1e-8));
    g.map(map_ipj());
    hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(e.index.i * 10 + e.index.j).margin(1e-8));
    Grid2<float> gg;
    gg = g;
    REQUIRE(g.resolution() == gg.resolution());
    REQUIRE(g.spacing() == gg.spacing());
    REQUIRE(g.origin() == gg.origin());
    g = 7.3f;
    hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(7.3).margin(1e-8));
    Array2<float> a(size2(10, 10));
    a = 1.f;
    g = a;
    hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(1).margin(1e-8));
  }
}
