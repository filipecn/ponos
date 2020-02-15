#include <catch2/catch.hpp>

#include <ponos/ponos.h>

using namespace ponos;

TEST_CASE("BBox3-sanity", "[geometry][bbox][access]") {
  auto l = point3(0, 1, 2);
  auto u = point3(3, 4, 3);
  bbox3 b(l, u);
  REQUIRE(b.lower == l);
  REQUIRE(b.upper == u);
  REQUIRE(b[0] == l);
  REQUIRE(b[1] == u);
  REQUIRE(b.corner(0) == point3(0, 1, 2));
  REQUIRE(b.corner(1) == point3(3, 1, 2));
  REQUIRE(b.corner(2) == point3(0, 4, 2));
  REQUIRE(b.corner(3) == point3(3, 4, 2));
  REQUIRE(b.corner(4) == point3(0, 1, 3));
  REQUIRE(b.corner(5) == point3(3, 1, 3));
  REQUIRE(b.corner(6) == point3(0, 4, 3));
  REQUIRE(b.corner(7) == point3(3, 4, 3));
}

TEST_CASE("BBox3-union", "[geometry][bbox][methods]") {
  {
    bbox3 a(point3(), point3(1));
    bbox3 b(point3(-1), point3());
    bbox3 c = make_union(a, b);
    point3 l(-1), u(1);
    REQUIRE(c.lower == l);
    REQUIRE(c.upper == u);
  }
}