#include <catch2/catch.hpp>

#include <ponos/ponos.h>

using namespace ponos;

TEST_CASE("Interpolation", "[numeric][interpolation]") {

  SECTION("linear") {
    { // 1D
      float dx = 0.01;
      auto f = [](float x) -> float { return cos(x) * sin(x); };
      HaltonSequence sampler;
      for (int i = 0; i < 1000; ++i) {
        auto p = sampler.randomFloat();
        REQUIRE(lerp<float>(p, f(0), f(dx)) == Approx(f(p * dx)).margin(1e-6));
      }
    }
    { // 2D
      auto f = [](float x, float y) -> float { return cos(x) * sin(y); };
      RNGSampler sampler;
      float dx = 0.01;
      for (int i = 0; i < 1000; ++i) {
        auto p = sampler.sample(bbox2::unitBox());
        REQUIRE(bilerp<float>(p.x, p.y, f(0.00, 0.00), f(dx, 0.00), f(dx, dx),
                              f(0.00, dx)) ==
                Approx(f(p.x * dx, p.y * dx)).margin(1e-6));
      }
      { // 3D
        // TODO
      }
    }
  }

  SECTION("monotonicCubic") {
    { // 1D test
      float dx = 0.01;
      auto f = [](float x) -> float { return cos(x) * sin(x); };
      for (float s = 0.0; s <= 1.0; s += 0.01) {
        REQUIRE(monotonicCubicInterpolate(f(-1 * dx), f(0), f(1 * dx),
                                          f(2 * dx),
                                          s) == Approx(f(s * dx)).margin(1e-7));
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
        REQUIRE(monotonicCubicInterpolate(v, point2(p.x, p.y)) ==
                Approx(f(dx + p.x * dx, dx + p.y * dx)).margin(1e-7));
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
        REQUIRE(monotonicCubicInterpolate(v, p) ==
                Approx(f(dx + p.x * dx, dx + p.y * dx, dx + p.z * dx))
                    .margin(1e-7));
      }
    }
  }
}

TEST_CASE("Grid", "[numeric][grid]") {
  SECTION("2d") {
    Grid2<float> g(size2(10, 10), vec2(0.1, 0.1), point2(1, 2));
    for (auto e : g.accessor())
      e.value = e.index.i * 10 + e.index.j;
    Grid2<float> gg;
    gg = g;
    REQUIRE(g.resolution() == gg.resolution());
    REQUIRE(g.spacing() == gg.spacing());
    REQUIRE(g.origin() == gg.origin());
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(gg.accessor()[e.index]).margin(1e-8));
    g = 7.3f;
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(7.3).margin(1e-8));
    Array2<float> a(size2(10, 10));
    a = 1.f;
    g = a;
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(1).margin(1e-8));
    gg = g;
    for (auto e : gg.accessor())
      REQUIRE(e.value == Approx(1).margin(1e-8));
  }
}

TEST_CASE("GridIterator", "[numeric][grid][iterator]") {

  SECTION("2d") {
    Grid2<float> g(size2(10, 10));
    g.setSpacing(vec2(0.1, 0.2));
    for (auto e : g.accessor()) {
      REQUIRE(e.worldPosition().x == Approx(e.index.i * 0.1).margin(1e-7));
      REQUIRE(e.worldPosition().y == Approx(e.index.j * 0.2).margin(1e-7));
      REQUIRE(e.region().size(0) == Approx(0.1).margin(1e-7));
      REQUIRE(e.region().size(1) == Approx(0.2).margin(1e-7));
      e.value = e.index.i * 10 + e.index.j;
    }
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(e.index.i * 10 + e.index.j).margin(1e-8));
  }
}

TEST_CASE("GridAccessor", "[numeric][grid][accessor]") {
  SECTION("2d") {
    float dx = 0.01;
    Grid2<float> g(size2(10, 10), vec2(dx, dx));
    auto f = [](point2 p) -> float { return std::sin(p.x) * std::cos(p.y); };
    for (auto ij : Index2Range<i32>(g.resolution())) {
      g.accessor()[ij] = f(g.accessor().worldPosition(ij));
      REQUIRE(g.accessor()[ij] ==
              Approx(f(g.accessor().worldPosition(ij))).margin(1e-8));
    }
    // cell info
    REQUIRE(g.accessor().cellRegion(index2(4, 4)).extends().x ==
            Approx(dx).margin(1e-8));
    REQUIRE(g.accessor().cellRegion(index2(4, 4)).extends().y ==
            Approx(dx).margin(1e-8));
    REQUIRE(g.accessor().cellIndex(point2(5.5 * dx, 7.8 * dx)) == index2(5, 7));
    REQUIRE(g.accessor().cellPosition(point2(3.1 * dx, 8.7 * dx)).x ==
            Approx(0.1).margin(1e-6));
    REQUIRE(g.accessor().cellPosition(point2(3.1 * dx, 8.7 * dx)).y ==
            Approx(0.7).margin(1e-6));
    SECTION("CLAMP_TO_EDGE + LINEAR INTERPOLATION") {
      auto acc =
          g.accessor(AddressMode::CLAMP_TO_EDGE, InterpolationMode::LINEAR);
      REQUIRE(acc[index2(-5, 5)] ==
              Approx(f(acc.worldPosition(index2(0, 5)))).margin(1e-8));
      REQUIRE(acc[index2(5, -5)] ==
              Approx(f(acc.worldPosition(index2(5, 0)))).margin(1e-8));
      REQUIRE(acc[index2(11, 5)] ==
              Approx(f(acc.worldPosition(index2(9, 5)))).margin(1e-8));
      REQUIRE(acc[index2(5, 11)] ==
              Approx(f(acc.worldPosition(index2(5, 9)))).margin(1e-8));
      for (index2 ij : Index2Range<i32>(size2(9, 9))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-5));
        }
      }
    }
    SECTION("CLAMP_TO_EDGE + MONOTONIC CUBIC INTERPOLATION") {
      auto acc = g.accessor(AddressMode::CLAMP_TO_EDGE,
                            InterpolationMode::MONOTONIC_CUBIC);
      for (index2 ij : Index2Range<i32>(index2(1, 1), index2(8, 8))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-7));
        }
      }
    }
    SECTION("BORDER + LINEAR INTERPOLATION") {
      auto acc = g.accessor(AddressMode::BORDER, InterpolationMode::LINEAR);
      REQUIRE(acc[index2(-5, 5)] == Approx(0).margin(1e-8));
      REQUIRE(acc[index2(5, -5)] == Approx(0).margin(1e-8));
      REQUIRE(acc[index2(11, 5)] == Approx(0).margin(1e-8));
      REQUIRE(acc[index2(5, 11)] == Approx(0).margin(1e-8));
      for (index2 ij : Index2Range<i32>(size2(9, 9))) {
        RNGSampler sampler;
        for (int i = 0; i < 1; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-5));
        }
      }
    }
    SECTION("BORDER + MONOTONIC CUBIC INTERPOLATION") {
      auto acc =
          g.accessor(AddressMode::BORDER, InterpolationMode::MONOTONIC_CUBIC);
      for (index2 ij : Index2Range<i32>(index2(1, 1), index2(8, 8))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-7));
        }
      }
    }
  }
}

TEST_CASE("VectorGrid", "[numeric][grid]") {
  SECTION("2d") {
    SECTION("CELL CENTERED") {
      VectorGrid2<float> vg(size2(10, 10), vec2(1));
      REQUIRE(vg.resolution() == size2(10, 10));
      REQUIRE(vg.u().resolution() == size2(10, 10));
      REQUIRE(vg.v().resolution() == size2(10, 10));
      REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg.u().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().y == Approx(0).margin(1e-8));
      VectorGrid2<float> vg2;
      vg2 = vg;
      REQUIRE(vg2.resolution() == size2(10, 10));
      REQUIRE(vg2.u().resolution() == size2(10, 10));
      REQUIRE(vg2.v().resolution() == size2(10, 10));
      REQUIRE(vg2.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg2.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg2.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg2.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg2.u().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg2.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg2.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg2.v().origin().y == Approx(0).margin(1e-8));
      vg.setGridType(VectorGridType::STAGGERED);
      REQUIRE(vg.resolution() == size2(10, 10));
      REQUIRE(vg.u().resolution() == size2(11, 10));
      REQUIRE(vg.v().resolution() == size2(10, 11));
      REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg.u().origin().x == Approx(-0.5).margin(1e-8));
      REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().y == Approx(-0.5).margin(1e-8));
    }
    SECTION("STAGGERED") {
      VectorGrid2<float> vg(VectorGridType::STAGGERED);
      vg.setResolution(size2(10, 10));
      REQUIRE(vg.resolution() == size2(10, 10));
      REQUIRE(vg.u().resolution() == size2(11, 10));
      REQUIRE(vg.v().resolution() == size2(10, 11));
      REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg.u().origin().x == Approx(-0.5).margin(1e-8));
      REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().y == Approx(-0.5).margin(1e-8));
      VectorGrid2<float> vg2;
      vg2 = vg;
      REQUIRE(vg2.resolution() == size2(10, 10));
      REQUIRE(vg2.u().resolution() == size2(11, 10));
      REQUIRE(vg2.v().resolution() == size2(10, 11));
      REQUIRE(vg2.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg2.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg2.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg2.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg2.u().origin().x == Approx(-0.5).margin(1e-8));
      REQUIRE(vg2.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg2.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg2.v().origin().y == Approx(-0.5).margin(1e-8));
      vg.setGridType(VectorGridType::CELL_CENTERED);
      REQUIRE(vg.resolution() == size2(10, 10));
      REQUIRE(vg.u().resolution() == size2(10, 10));
      REQUIRE(vg.v().resolution() == size2(10, 10));
      REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg.u().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().y == Approx(0).margin(1e-8));
    }
  }
}

TEST_CASE("VectorGridAccessor", "[numeric][grid][accessor]") {
  auto f = [](point2 wp) -> float { return sin(wp.x) * cos(wp.y); };
  SECTION("2d") {
    SECTION("CELL CENTERED") {
      VectorGrid2<float> vg(size2(10), vec2(0.1));
      for (auto e : vg.u().accessor()) {
        REQUIRE(e.worldPosition().x == Approx(0.1 * e.index.i).margin(1e-8));
        REQUIRE(e.worldPosition().y == Approx(0.1 * e.index.j).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      for (auto e : vg.v().accessor()) {
        REQUIRE(e.worldPosition().x == Approx(0.1 * e.index.i).margin(1e-8));
        REQUIRE(e.worldPosition().y == Approx(0.1 * e.index.j).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      auto acc = vg.accessor();
      for (index2 ij : Index2Range<i32>(vg.resolution())) {
        auto v = acc[ij];
        REQUIRE(v.x == Approx(f(acc.worldPosition(ij))));
        REQUIRE(v.y == Approx(f(acc.worldPosition(ij))));
      }
    }
    SECTION("STAGGERED") {
      VectorGrid2<float> vg(size2(10), vec2(0.1), point2(),
                            VectorGridType::STAGGERED);
      for (auto e : vg.u().accessor()) {
        REQUIRE(e.worldPosition().x ==
                Approx(0.1 * e.index.i - 0.05).margin(1e-8));
        REQUIRE(e.worldPosition().y == Approx(0.1 * e.index.j).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      for (auto e : vg.v().accessor()) {
        REQUIRE(e.worldPosition().x == Approx(0.1 * e.index.i).margin(1e-8));
        REQUIRE(e.worldPosition().y ==
                Approx(0.1 * e.index.j - 0.05).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      auto acc = vg.accessor();
      for (index2 ij : Index2Range<i32>(vg.resolution())) {
        auto v = acc[ij];
        REQUIRE(v.x == Approx((f(acc.u().worldPosition(ij)) +
                               f(acc.u().worldPosition(ij.right()))) /
                              2));
        REQUIRE(v.y == Approx((f(acc.v().worldPosition(ij)) +
                               f(acc.v().worldPosition(ij.up()))) /
                              2));
      }
    }
  }
}