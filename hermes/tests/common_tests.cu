#include <catch2/catch.hpp>

#include <hermes/hermes.h>

using namespace hermes::cuda;

struct sum {};

TEST_CASE("Reduce", "[common][reduce]") {
  SECTION("1-dimension single") {
    SECTION("ReducePredicates") {
      Array1<i32> a(1000, 10);
      REQUIRE(reduce<i32, i32, ReducePredicates::sum<i32>>(
                  a, ReducePredicates::sum<i32>()) == 1000 * 10);
      REQUIRE(reduce<i32, u8, ReducePredicates::is_equal_to_value<i32>>(
                  a, ReducePredicates::is_equal_to_value<i32>(10)) == true);
      std::vector<i32> data(1000);
      for (int i = 0; i < 1000; ++i)
        data[i] = i - 500;
      Array1<i32> b = data;
      REQUIRE(reduce<i32, i32, ReducePredicates::min<i32>>(
                  b, ReducePredicates::min<i32>()) == -500);
      REQUIRE(reduce<i32, i32, ReducePredicates::min_abs<i32>>(
                  b, ReducePredicates::min_abs<i32>()) == 0);
      REQUIRE(reduce<i32, i32, ReducePredicates::max<i32>>(
                  b, ReducePredicates::max<i32>()) == 499);
      REQUIRE(reduce<i32, i32, ReducePredicates::max_abs<i32>>(
                  b, ReducePredicates::max_abs<i32>()) == 500);
    }
  }
  SECTION("1-dimension double") {
    SECTION("ReducePredicates") {
      Array1<i32> a(1000, 10), aa(1000, 10), b(1000, -20);
      REQUIRE(reduce<i32, i32, ReducePredicates::sum<i32>>(
                  a, b, ReducePredicates::sum<i32>()) == -10 * 1000);
      REQUIRE(reduce<i32, u8, ReducePredicates::is_equal<i32>>(
                  a, a, ReducePredicates::is_equal<i32>()) == true);
      REQUIRE(reduce<i32, i32, ReducePredicates::min<i32>>(
                  a, b, ReducePredicates::min<i32>()) == -20);
      REQUIRE(reduce<i32, i32, ReducePredicates::min_abs<i32>>(
                  a, b, ReducePredicates::min_abs<i32>()) == 10);
      REQUIRE(reduce<i32, i32, ReducePredicates::max<i32>>(
                  a, b, ReducePredicates::max<i32>()) == 10);
      REQUIRE(reduce<i32, i32, ReducePredicates::max_abs<i32>>(
                  a, b, ReducePredicates::max_abs<i32>()) == 20);
    }
  }
  SECTION("2-dimension") {
    SECTION("ReducePredicates") {
      Array2<i32> a(size2(1000), 10);
      REQUIRE(reduce<i32, i32, ReducePredicates::sum<i32>>(
                  a, ReducePredicates::sum<i32>()) == 1000 * 1000 * 10);
      REQUIRE(reduce<i32, u8, ReducePredicates::is_equal_to_value<i32>>(
                  a, ReducePredicates::is_equal_to_value<i32>(10)) == true);
      ponos::Array2<i32> data(ponos::size2(1000));
      for (auto e : data)
        e.value = e.index.i * e.index.j - 500000;
      Array2<i32> b = data;
      CHECK_CUDA(cudaDeviceSynchronize());
      REQUIRE(reduce<i32, i32, ReducePredicates::min<i32>>(
                  b, ReducePredicates::min<i32>()) == -500000);
      REQUIRE(reduce<i32, i32, ReducePredicates::min_abs<i32>>(
                  b, ReducePredicates::min_abs<i32>()) == 0);
      REQUIRE(reduce<i32, i32, ReducePredicates::max<i32>>(
                  b, ReducePredicates::max<i32>()) == 999 * 999 - 500000);
      REQUIRE(reduce<i32, i32, ReducePredicates::max_abs<i32>>(
                  b, ReducePredicates::max_abs<i32>()) == 500000);
    }
  }
}