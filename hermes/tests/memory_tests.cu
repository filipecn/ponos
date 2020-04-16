#include <catch2/catch.hpp>

#include <hermes/hermes.h>

using namespace hermes::cuda;

texture<f32, cudaTextureType2D> tex2;

__global__ void __copyArray(Array2Accessor<f32> dst) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (dst.contains(index))
    dst[index] = tex2D(tex2, index.i, index.j);
}

TEST_CASE("Array-access", "[memory][array][access]") {
  print_cuda_devices();
  {
    Array1<vec2> a(1000);
    REQUIRE(a.size() == 1000u);
    // CUDA_MEMORY_USAGE;
  }
  {
    Array1<index2> a(1000, index2(1, 3));
    auto v = a.hostData();
    for (auto vv : v)
      REQUIRE(vv == index2(1, 3));
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

TEST_CASE("Array", "[memory][array][access]") {
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

TEST_CASE("CuArray", "[memory][cuarray]") {
  SECTION("2d") {
    ponos::Array2<f32> data(ponos::size2(128));
    for (auto ij : ponos::Index2Range<i32>(data.size()))
      data[ij] = ij.i * 100 + ij.j;
    Array2<f32> d_data = data;
    CuArray2<f32> c = d_data;
    CuArray2<f32> c2 = data;
    auto td = ThreadArrayDistributionInfo(d_data.size());
    tex2.normalized = 0;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<f32>();
    { // from c
      Array2<f32> r(d_data.size());
      CHECK_CUDA(cudaBindTextureToArray(tex2, c.data(), channelDesc));
      __copyArray<<<td.gridSize, td.blockSize>>>(r.accessor());
      auto h = r.hostData();
      for (auto ij : ponos::Index2Range<i32>(data.size()))
        REQUIRE(h[ij] == Approx(ij.i * 100 + ij.j).margin(1e-6));
    }
    { // from c2
      Array2<f32> r(d_data.size());
      CHECK_CUDA(cudaBindTextureToArray(tex2, c2.data(), channelDesc));
      __copyArray<<<td.gridSize, td.blockSize>>>(r.accessor());
      auto h = r.hostData();
      for (auto ij : ponos::Index2Range<i32>(data.size()))
        REQUIRE(h[ij] == Approx(ij.i * 100 + ij.j).margin(1e-6));
    }
    cudaUnbindTexture(tex2);
  }
}