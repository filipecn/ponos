#include <catch2/catch.hpp>

#include <ponos/ponos.h>

using namespace ponos;

TEST_CASE("Index-sanity", "[common][index][access]") {
  { // Index2
    index2 a;
    index2 b;
    REQUIRE(a == b);
    b.j = 1;
    REQUIRE(a != b);
  }
  { // Index2Range
    int cur = 0;
    for (auto index : Index2Range<i32>(10, 10)) {
      REQUIRE(cur % 10 == index.i);
      REQUIRE(cur / 10 == index.j);
      cur++;
    }
    REQUIRE(cur == 10 * 10);
  }
  { // Index3
    index3 a;
    index3 b;
    REQUIRE(a == b);
    b.j = 1;
    REQUIRE(a != b);
  }
  { // Index3Range
    int cur = 0;
    for (auto index : Index3Range<i32>(10, 10, 10)) {
      REQUIRE((cur % 100) % 10 == index.i);
      REQUIRE((cur % 100) / 10 == index.j);
      REQUIRE(cur / 100 == index.k);
      cur++;
    }
    REQUIRE(cur == 10 * 10 * 10);
  }
}