#include <gtest/gtest.h>
#include <hermes/hermes.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;

TEST(Blas, dot) {
  thrust::host_vector<int> h_a(1000, 1);
  thrust::host_vector<int> h_b(1000, 2);
  thrust::device_vector<int> d_a(h_a);
  thrust::device_vector<int> d_b(h_b);
  int d = dot(thrust::raw_pointer_cast(d_a.data()),
              thrust::raw_pointer_cast(d_b.data()), 1000);
  EXPECT_EQ(d, 2000);
}

TEST(Blas, axpy) {
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
