#include <catch2/catch.hpp>

#include <hermes/hermes.h>

using namespace hermes::cuda;

TEST_CASE("Array-access", "[memory][array][access]") {
  print_cuda_devices();
  {
    Array1<vec2> a(1000);
    REQUIRE(a.size() == 1000u);
  }
  {
    Array1<vec2i> a(1000, vec2i(1, 3));
    auto v = a.hostData();
    for (auto vv : v)
      REQUIRE(vv == vec2i(1, 3));
  }
  {
    Array1<char> a(1000);
    a = 'a';
    auto v = a.hostData();
    for (auto c : v)
      REQUIRE(c == 'a');
  }
  {
    std::vector<int> data(1000);
    for (int i = 0; i < 1000; ++i)
      data[i] = i;
    Array1<int> a = data;
    auto v = a.hostData();
    for (int i = 0; i < 1000; ++i)
      REQUIRE(v[i] == i);
  }
  {
    Array1<int> a = std::move(Array1<int>(1000));
    Array1<int> b(Array1<int>(1000));
    b = a;
    a = std::vector<int>(1000);
  }
}

TEST_CASE("Array-sanity", "[memory][array][access]") {
  SECTION("2d") {
    {
      Array2<vec2> a(size2(10, 10));
      REQUIRE(a.size() == size2(10, 10));
      REQUIRE(a.memorySize() == 10 * a.pitch());
      Array2<vec2> b = a;
      REQUIRE(b.size() == size2(10, 10));
      REQUIRE(b.memorySize() == 10 * b.pitch());
    }
    {
      Array2<int> a(size2(10, 10));
      a = 3;
      auto ha = a.hostData();
      int count = 0;
      for (auto e : ha) {
        REQUIRE(e.value == 3);
        count++;
      }
      REQUIRE(count == 10 * 10);
    }
    {
      std::vector<Array2<int>> v;
      v.emplace_back(size2(10, 10));
      v.emplace_back(size2(10, 10));
      v.emplace_back(size2(10, 10));
      std::vector<ponos::Array2<int>> h_v;
      h_v.emplace_back(ponos::size2(10, 10));
      h_v.emplace_back(ponos::size2(10, 10));
      h_v.emplace_back(ponos::size2(10, 10));
      for (int i = 0; i < 3; i++) {
        for (auto e : h_v[i])
          e.value = e.index.i * 10 + e.index.j;
        v[i] = h_v[i];
      }
      std::vector<Array2<int>> vv;
      for (auto &e : v)
        vv.emplace_back(e);
      for (int i = 0; i < 3; i++) {
        auto h = vv[i].hostData();
        int count = 0;
        for (auto hh : h) {
          REQUIRE(hh.value == hh.index.i * 10 + hh.index.j);
          count++;
        }
        REQUIRE(count == 100);
      }
    }
    {
      Array2<int> a = std::move(Array2<int>(size2(10, 10)));
      Array2<int> b(Array2<int>(size2(10, 10)));
    }
  }
}

struct map_ipj {
  __device__ void operator()(index2 index, int &value) const {
    value = index.i + index.j;
  }
};

TEST_CASE("Array-Methods", "[memory][array][methods]") {
  {
    array2i a(size2(10));
    a.map(map_ipj());
    auto ha = a.hostData();
    for (auto e : ha)
      REQUIRE(e.value == e.index.i + e.index.j);
  }
}
