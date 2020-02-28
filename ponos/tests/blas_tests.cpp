#include <catch2/catch.hpp>

#include <ponos/ponos.h>

using namespace ponos;

TEST_CASE("Vector", "[blas][vector]") {
  SECTION("constructors") {
    Vector<f32> a;
    REQUIRE(a.size() == 0);
    Vector<f32> b(10);
    REQUIRE(b.size() == 10);
    Vector<f32> c(10, 3.f);
    for (int i = 0; i < 10; ++i)
      REQUIRE(c[i] == Approx(3.f).margin(1e-8));
    auto d = c;
    for (int i = 0; i < 10; ++i)
      REQUIRE(d[i] == Approx(3.f).margin(1e-8));
  }
  SECTION("operators") {
    Vector<f32> a(10, 0.f);
    a += 3.f;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(3.f).margin(1e-8));
    a -= 2.f;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(1.f).margin(1e-8));
    a *= 14.f;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(14.f).margin(1e-8));
    a /= 2.f;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(7.f).margin(1e-8));
    Vector<f32> b(10, 2.f);
    a *= b;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(14.f).margin(1e-8));
    a /= b;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(7.f).margin(1e-8));
    a += b;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(9.f).margin(1e-8));
    a -= b;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(7.f).margin(1e-8));
    a = a * 2.f;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(14.f).margin(1e-8));
    a = 2.f * a;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(28.f).margin(1e-8));
    a = 8.f / b;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(4.f).margin(1e-8));
    a = a / 2.f;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == Approx(2.f).margin(1e-8));
    REQUIRE(a == 2.f);
    REQUIRE(a == b);
    a = 3.f;
    REQUIRE(a == 3.f);
  }
}

TEST_CASE("Blas", "[blas]") {
  SECTION("dot") {
    Vector<f32> a(10);
    Vector<f32> b(10);
    f32 ans = 0.f;
    for (int i = 0; i < 10; ++i) {
      a[i] = i * 3;
      b[i] = -2 * i;
      ans += (i * 3) * (-2 * i);
    }
    REQUIRE(BLAS::dot(a, b) == Approx(ans).margin(1e-8));
  }
  SECTION("axpy") {
    Vector<f32> ans(10);
    Vector<f32> r(10);
    Vector<f32> x(10);
    Vector<f32> y(10);
    f32 a = 2.f;
    for (int i = 0; i < 10; ++i) {
      x[i] = -7 * i;
      y[i] = i;
      r[i] = a * x[i] + y[i];
    }
    BLAS::axpy(a, x, y, ans);
    REQUIRE(ans == r);
  }
  SECTION("infNorm") {
    Vector<f32> r(10);
    for (int i = 0; i < 10; ++i)
      r[i] = i - 7;
    REQUIRE(BLAS::infNorm(r) == Approx(7.f).margin(1e-8));
  }
}