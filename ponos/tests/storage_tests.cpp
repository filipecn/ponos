#include <catch2/catch.hpp>

#include <ponos/ponos.h>

using namespace ponos;

TEST_CASE("Array2-sanity", "[storage][array][access]") {
  {
    Array2<vec2> a(size2(10, 10));
    REQUIRE(a.pitch() == 10 * sizeof(vec2));
    REQUIRE(a.size() == size2(10, 10));
    REQUIRE(a.memorySize() == 10 * 10 * sizeof(vec2));
    for (index2 ij : Index2Range<i32>(a.size()))
      a[ij] = vec2(ij.i, ij.j);
    Array2<vec2> b = a;
    for (index2 ij : Index2Range<i32>(a.size()))
      REQUIRE(a[ij] == b[ij]);
  }
  {
    Array2<int> a(size2(10, 10));
    a = 3;
    int count = 0;
    for (index2 ij : Index2Range<i32>(a.size()))
      REQUIRE(a[ij] == 3);
    for (const auto &e : a) {
      REQUIRE(e.value == 3);
      count++;
    }
    std::cerr << a << std::endl;
    REQUIRE(count == 10 * 10);
  }
  {
    std::vector<Array2<int>> v;
    v.emplace_back(size2(10, 10));
    v.emplace_back(size2(10, 10));
    v.emplace_back(size2(10, 10));
    for (int i = 0; i < 3; i++)
      for (index2 ij : Index2Range<i32>(v[i].size()))
        v[i][ij] = ij.i * 10 + ij.j;
    std::vector<Array2<int>> vv = v;
    for (int i = 0; i < 3; i++)
      for (index2 ij : Index2Range<i32>(v[i].size()))
        REQUIRE(vv[i][ij] == ij.i * 10 + ij.j);
  }
  {
    Array2<int> a = std::move(Array2<int>(size2(10, 10)));
    Array2<int> b(Array2<int>(size2(10, 10)));
  }
}

TEST_CASE("Array2-iterator", "[storage][array][iterator]") {
  {
    Array2<vec2> a(size2(10, 10));
    for (auto e : a)
      e.value = vec2(1, 2);
    int count = 0;
    for (auto e : a) {
      count++;
      REQUIRE(e.value == vec2(1, 2));
    }
    REQUIRE(count == 100);
  }
}