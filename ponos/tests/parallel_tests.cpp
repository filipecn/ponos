//
// Created by filipecn on 25/02/2021.
//
#include <catch2/catch.hpp>

#include <ponos/ponos.h>

using namespace ponos;

TEST_CASE("Parallel", "[parallel]") {
  SECTION("loop") {
//    std::vector<int> v(100000, 0);
//    Parallel::loop([&](int i) {
//      v[i] = 1;
//    }, v.size(), 1);
//    auto sum = 0;
//    for (auto i : v)
//      sum += i;
//    REQUIRE(sum == v.size());
  }//
  SECTION("loop_range") {
    std::atomic<int> sum = 0;
    u64 n = 10000000;
    Parallel::loop(0, n, [&](u64 f, u64 l) {
      u64 t = 0;
      for (u64 i = f; i <= l; ++i)
        t++;
      sum += t;
    }, 1000);
    REQUIRE(sum == n);
  }
}