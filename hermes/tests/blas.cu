#include <gtest/gtest.h>
#include <hermes/hermes.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;

TEST(Blas, dot) {
  {
    thrust::host_vector<int> h_a(1000, 1);
    thrust::host_vector<int> h_b(1000, 2);
    thrust::device_vector<int> d_a(h_a);
    thrust::device_vector<int> d_b(h_b);
    int d = dot(thrust::raw_pointer_cast(d_a.data()),
                thrust::raw_pointer_cast(d_b.data()), 1000);
    EXPECT_EQ(d, 2000);
  }
  {
    MemoryBlock1Di aux, d_a(1000, 1);
    MemoryBlock1Di d_b(1000, 2);
    int d = dot(d_a, d_b, aux);
    EXPECT_EQ(d, 2000);
  }
}

TEST(Blas, sub) {
  {
    MemoryBlock1Di d_a(1000, 1);
    MemoryBlock1Di d_b(1000, 2), d_r(1000, 0);
    sub(d_a, d_b, d_r);
    MemoryBlock1Hi r(1000, 0);
    memcpy(r, d_r);
    auto acc = r.accessor();
    for (int i = 0; i < 1000; i++)
      EXPECT_EQ(acc[i], -1);
  }
}

TEST(Blas, axpy) {
  {
    thrust::host_vector<float> h_x(1000, 3);
    thrust::host_vector<float> h_y(1000, 4);
    thrust::device_vector<float> d_x(h_x);
    thrust::device_vector<float> d_y(h_y);
    thrust::device_vector<float> d_r(1000);
    axpy(0.4f, thrust::raw_pointer_cast(d_x.data()),
         thrust::raw_pointer_cast(d_y.data()),
         thrust::raw_pointer_cast(d_r.data()), 1000);
    thrust::host_vector<float> h_r(d_r);
    for (int i = 0; i < 1000; i++)
      EXPECT_EQ(h_r[i], 0.4f * h_x[i] + h_y[i]);
  }
  {
    MemoryBlock1Df aux, d_x(1000, 3);
    MemoryBlock1Df d_y(1000, 4);
    MemoryBlock1Df d_r(1000, 0);
    axpy(4.f, d_x, d_y, d_r);
    MemoryBlock1Hf h_r(1000, 0);
    memcpy(h_r, d_r);
    auto acc = h_r.accessor();
    for (int i = 0; i < 1000; i++)
      EXPECT_EQ(acc[i], 16.f);
  }
}
