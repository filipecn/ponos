#include <catch2/catch.hpp>

#include <hermes/hermes.h>

using namespace hermes::cuda;

///////////////////////////////////////////////////////////////////////////////
////////////////////       INTERPOLATION     //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
////////////////////       GRID              //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct map_ipj {
  __device__ void operator()(index2 index, float &value) const {
    value = index.i * 10 + index.j;
  }
};

TEST_CASE("Grid", "[numeric][grid][access]") {
  SECTION("2d") {
    // check paramters constructor
    Grid2<float> g(size2(10, 10), vec2(0.1, 0.1), point2(1, 2));
    REQUIRE(g.spacing().x == Approx(0.1).margin(1e-8));
    REQUIRE(g.spacing().y == Approx(0.1).margin(1e-8));
    REQUIRE(g.origin().x == Approx(1).margin(1e-8));
    REQUIRE(g.origin().y == Approx(2).margin(1e-8));
    // value assign operator
    g = 3.0;
    // check host data method
    ponos::Grid2<float> hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(3).margin(1e-8));
    // check map method
    g.map(map_ipj());
    hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(e.index.i * 10 + e.index.j).margin(1e-8));
    // default constructor
    Grid2<float> gg;
    // check assign operator
    gg = g;
    REQUIRE(g.resolution() == gg.resolution());
    REQUIRE(g.spacing() == gg.spacing());
    REQUIRE(g.origin() == gg.origin());
    g = 7.3f;
    hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(7.3).margin(1e-8));
    // host data assign operator
    Array2<float> a(size2(10, 10));
    a = 1.f;
    g = a;
    hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(1).margin(1e-8));
    // check copy constructors
    std::vector<Grid2<f32>> grid_vector;
    grid_vector.emplace_back();
    grid_vector[0].setResolution(size2(10, 10));
    grid_vector.emplace_back();
    grid_vector.emplace_back();
    REQUIRE(grid_vector[0].resolution() == size2(10));
    REQUIRE(grid_vector[1].resolution() == size2(0));
    // check constructor from host grid
    Grid2<float> d_g(hg);
    REQUIRE(d_g.resolution() == size2(hg.resolution()));
    REQUIRE(d_g.spacing() == vec2(hg.spacing()));
    REQUIRE(d_g.origin() == point2(hg.origin()));
  }
}

TEST_CASE("VectorGrid", "[numeric][grid]") {
  SECTION("2d") {
    SECTION("constructors operatiors") {
      VectorGrid2<float> empty;
      REQUIRE(empty.resolution() == size2(0, 0));
      empty = VectorGrid2<float>(size2(10, 10), vec2f(1));
      REQUIRE(empty.resolution() == size2(10, 10));
      VectorGrid2<float> g = empty;
      REQUIRE(g.resolution() == size2(10, 10));
      auto g2 = std::move(VectorGrid2<float>(size2(7, 7), vec2f(1)));
      REQUIRE(g2.resolution() == size2(7, 7));
      std::vector<VectorGrid2<float>> gs;
      gs.emplace_back(size2(10, 10), vec2(1));
      gs.emplace_back(size2(7, 7), vec2(1));
      std::vector<VectorGrid2<float>> gs2 = gs;
    }
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
      vg.setGridType(ponos::VectorGridType::STAGGERED);
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
      VectorGrid2<float> vg(ponos::VectorGridType::STAGGERED);
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
      vg.setGridType(ponos::VectorGridType::CELL_CENTERED);
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

struct map_sincos {
  map_sincos(Grid2Accessor<float> acc) : acc(acc) {}
  __device__ void operator()(index2 index, float &value) const {
    auto wp = acc.worldPosition(index);
    value = sin(wp.x) * cos(wp.y);
  }
  Grid2Accessor<float> acc;
};

TEST_CASE("VectorGridAccessor", "[numeric][grid][accessor]") {
  auto f = [](ponos::point2 wp) -> float { return sin(wp.x) * cos(wp.y); };
  SECTION("2d") {
    SECTION("methods") {
      ponos::VectorGrid2<float> hg(ponos::size2(10), ponos::vec2(0.1),
                                   ponos::point2(),
                                   ponos::VectorGridType::STAGGERED);
      VectorGrid2<float> dg(size2(10), vec2(0.1), point2(),
                            ponos::VectorGridType::STAGGERED);
      auto hacc = hg.accessor();
      auto dacc = dg.accessor();
      for (index2 ij : Index2Range<i32>(dg.resolution())) {
        REQUIRE(hacc.u().worldPosition(ij.ponos()).x ==
                Approx(dacc.u().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.u().worldPosition(ij.ponos()).y ==
                Approx(dacc.u().worldPosition(ij).y).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).x ==
                Approx(dacc.v().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).y ==
                Approx(dacc.v().worldPosition(ij).y).margin(1e-8));
      }
    }
    SECTION("device to host") {
      VectorGrid2<float> dg(size2(10), vec2f(0.01));
      dg.u().map(map_sincos(dg.u().accessor()));
      dg.v().map(map_sincos(dg.v().accessor()));
      auto hg = dg.hostData();
      auto dacc = dg.accessor();
      auto hacc = hg.accessor();
      for (index2 ij : Index2Range<i32>(dg.resolution())) {
        auto v = hacc[ij.ponos()];
        REQUIRE(v.x == Approx(f(hacc.worldPosition(ij.ponos()))));
        REQUIRE(v.y == Approx(f(hacc.worldPosition(ij.ponos()))));
        REQUIRE(hacc.u().worldPosition(ij.ponos()).x ==
                Approx(dacc.u().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.u().worldPosition(ij.ponos()).y ==
                Approx(dacc.u().worldPosition(ij).y).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).x ==
                Approx(dacc.v().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).y ==
                Approx(dacc.v().worldPosition(ij).y).margin(1e-8));
      }
    }
    SECTION("host to device") {
      ponos::VectorGrid2<float> hg(ponos::size2(10), ponos::vec2(0.1),
                                   ponos::point2(),
                                   ponos::VectorGridType::STAGGERED);
      for (auto e : hg.u().accessor())
        e.value = f(e.worldPosition());
      for (auto e : hg.v().accessor())
        e.value = f(e.worldPosition());

      VectorGrid2<float> dg = hg;

      auto hdg = dg.hostData();
      auto hacc = hg.accessor();
      auto dacc = dg.accessor();
      for (index2 ij : Index2Range<i32>(dg.resolution())) {
        auto v = hacc[ij.ponos()];
        REQUIRE(v.x == Approx((f(hacc.u().worldPosition(ij.ponos())) +
                               f(hacc.u().worldPosition(ij.right().ponos()))) /
                              2));
        REQUIRE(v.y == Approx((f(hacc.v().worldPosition(ij.ponos())) +
                               f(hacc.v().worldPosition(ij.up().ponos()))) /
                              2));
        REQUIRE(hacc.u().worldPosition(ij.ponos()).x ==
                Approx(dacc.u().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.u().worldPosition(ij.ponos()).y ==
                Approx(dacc.u().worldPosition(ij).y).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).x ==
                Approx(dacc.v().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).y ==
                Approx(dacc.v().worldPosition(ij).y).margin(1e-8));
      }
    }
  }
}

TEST_CASE("FDMatrix", "[numeric][fdmatrix]") {
  SECTION("2d") {
    SECTION("sanity") {
      // y
      // |
      //  ---x
      //  S S S S  - - - -
      //  S S S S  - - - -
      //  S S S S  - - - -
      //  S S S S  - - - -
      ponos::size2 size(4);
      ponos::FDMatrix2<f32> A(size);
      auto &indices = A.indexData();
      int curIndex = 0;
      for (ponos::index2 ij : ponos::Index2Range<i32>(size))
        if (ij.j == 0 || ij.j == static_cast<i64>(size.height - 1) ||
            ij.i == 0 || ij.i == static_cast<i64>(size.width - 1))
          indices[ij] = -1;
        else
          indices[ij] = curIndex++;
      FDMatrix2<f32> d_A = A;
      auto h_A = d_A.hostData();
      for (ponos::index2 ij : ponos::Index2Range<i32>(
               ponos::index2(1), ponos::index2(size).plus(-1, -1))) {
        if (ij.i > 1)
          REQUIRE(h_A.elementIndex(ij.left()) != -1);
        else if (ij.i == 1)
          REQUIRE(h_A.elementIndex(ij.left()) == -1);
        if (ij.j > 1)
          REQUIRE(h_A.elementIndex(ij.down()) != -1);
        else if (ij.j == 1)
          REQUIRE(h_A.elementIndex(ij.down()) == -1);
        if (ij.i < static_cast<i64>(size.width) - 2)
          REQUIRE(h_A.elementIndex(ij.right()) != -1);
        else if (ij.i == static_cast<i64>(size.width) - 2)
          REQUIRE(h_A.elementIndex(ij.right()) == -1);
        if (ij.j < static_cast<i64>(size.height) - 2)
          REQUIRE(h_A.elementIndex(ij.up()) != -1);
        else if (ij.j == static_cast<i64>(size.height) - 2)
          REQUIRE(h_A.elementIndex(ij.up()) == -1);
        h_A(ij, ij) = 6;
        h_A(ij, ij.right()) = 1;
        h_A(ij, ij.up()) = 2;
      }
    }
    SECTION("matrix vector") {
      // 3 4 5 0  0     14
      // 4 3 0 5  1  =  18
      // 5 0 3 4  2     18
      // 0 5 4 3  3     22
      ponos::FDMatrix2<f32> A(ponos::size2(2));
      int index = 0;
      for (ponos::index2 ij : ponos::Index2Range<i32>(A.gridSize()))
        A.indexData()[ij] = index++;
      for (ponos::index2 ij : ponos::Index2Range<i32>(A.gridSize())) {
        A(ij, ij) = 3;
        A(ij, ij.right()) = 4;
        A(ij, ij.up()) = 5;
      }
      ponos::Vector<f32> x(A.size(), 0);
      for (u32 i = 0; i < x.size(); i++)
        x[i] = i;
      float ans[4] = {14, 18, 18, 22};
      FDMatrix2<f32> d_A = A;
      Vector<f32> d_x = x;
      Vector<f32> r = d_A * d_x;
      auto h_r = r.hostData();
      for (u32 i = 0; i < r.size(); i++)
        REQUIRE(h_r[i] == ans[i]);
    }
  }
}