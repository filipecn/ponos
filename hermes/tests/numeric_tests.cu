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
