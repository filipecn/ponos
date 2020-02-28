#include <catch2/catch.hpp>

#include <hermes/hermes.h>

using namespace hermes::cuda;

TEST_CASE("Vector", "[blas][vector]") {
  SECTION("constructors") {
    Vector<f32> a;
    REQUIRE(a.size() == 0);
    Vector<f32> b(10);
    REQUIRE(b.size() == 10);
    Vector<f32> c(10, 3.f);
    auto hc = c.hostData();
    for (int i = 0; i < 10; ++i)
      REQUIRE(hc[i] == Approx(3.f).margin(1e-8));
    auto d = c;
    auto hd = d.hostData();
    for (int i = 0; i < 10; ++i)
      REQUIRE(hd[i] == Approx(3.f).margin(1e-8));
  }
  SECTION("operators") {
    Vector<f32> a(10000, 0.f);
    a += 3.f;
    REQUIRE(a == 3.f);
    a -= 2.f;
    REQUIRE(a == 1.f);
    a *= 14.f;
    REQUIRE(a == 14.f);
    a /= 2.f;
    REQUIRE(a == 7.f);
    Vector<f32> b(10000, 2.f);
    a *= b;
    REQUIRE(a == 14.f);
    a /= b;
    REQUIRE(a == 7.f);
    a += b;
    REQUIRE(a == 9.f);
    a -= b;
    REQUIRE(a == 7.f);
    a = a * 2.f;
    REQUIRE(a == 14.f);
    a = 2.f * a;
    REQUIRE(a == 28.f);
    a = 8.f / b;
    REQUIRE(a == 4.f);
    a = a / 2.f;
    REQUIRE(a == 2.f);
    REQUIRE(a == b);
    a = 3.f;
    REQUIRE(a == 3.f);
  }
}

TEST_CASE("Blas", "[blas]") {
  SECTION("dot") {
    ponos::Vector<f32> ha(10);
    ponos::Vector<f32> hb(10);
    f32 ans = 0.f;
    for (int i = 0; i < 10; ++i) {
      ha[i] = i * 3;
      hb[i] = -2 * i;
      ans += (i * 3) * (-2 * i);
    }
    Vector<f32> a = ha;
    Vector<f32> b = hb;
    REQUIRE(BLAS::dot(a, b) == Approx(ans).margin(1e-8));
  }
  SECTION("axpy") {
    ponos::Vector<f32> h_ans(10);
    ponos::Vector<f32> h_r(10);
    ponos::Vector<f32> h_x(10);
    ponos::Vector<f32> h_y(10);
    f32 a = 2.f;
    for (int i = 0; i < 10; ++i) {
      h_x[i] = -7 * i;
      h_y[i] = i;
      h_r[i] = a * h_x[i] + h_y[i];
    }
    Vector<f32> ans = h_ans;
    Vector<f32> r = h_r;
    Vector<f32> x = h_x;
    Vector<f32> y = h_y;
    BLAS::axpy(a, x, y, ans);
    REQUIRE(ans == r);
  }
  SECTION("infNorm") {
    ponos::Vector<f32> h_r(10);
    for (int i = 0; i < 10; ++i)
      h_r[i] = i - 7;
    Vector<f32> r = h_r;
    REQUIRE(BLAS::infNorm(r) == Approx(7.f).margin(1e-8));
  }
}