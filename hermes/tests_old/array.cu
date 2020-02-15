#include <gtest/gtest.h>
#include <hermes/hermes.h>
#include <thrust/device_vector.h>

using namespace hermes;

template <typename T>
__global__ void __applyAnalyticFunction(const Vector<T, 3> *points, int n,
                                        double *f) {
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
    Vector<T, 3> p = points[index];
    f[index] = cos(5 * p.x()) * cos(5 * p.y()) * cos(5 * p.z());
  }
}

TEST(Array3, fill) {
  Vector3u size(128);
  thrust::device_vector<int> d_data(size[0] * size[1] * size[2], 0);
  Array3<int> array(size, thrust::raw_pointer_cast(d_data.data()));
  array.fill(12);
  thrust::host_vector<int> h_data(d_data);
  for (int i = 0; i < h_data.size(); i++)
    EXPECT_EQ(h_data[i], 12);
}