#include <hermes/storage/cuda_storage_utils.h>
#include <poseidon/simulation/cuda_smoke_injector.h>

texture<float, cudaTextureType2D> fieldTex2;

namespace poseidon {

namespace cuda {

__global__ void __injectCircle(float *out,
                               hermes::cuda::Transform2<float> toWorld,
                               hermes::cuda::point2f center, float radius2,
                               int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    // out[y * w + x] = 0;
    auto cp = toWorld(hermes::cuda::point2f(x, y));
    if ((cp - center).length2() <= radius2)
      out[y * w + x] = 1;
  }
}

__global__ void __injectSphere(hermes::cuda::RegularGrid3Accessor<float> acc,
                               hermes::cuda::point3f center, float radius2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < acc.resolution().x && y < acc.resolution().y &&
      z < acc.resolution().z) {
    auto cp = acc.worldPosition(x, y, z);
    acc(x, y, z) = 0;
    if ((cp - center).length2() <= radius2)
      acc(x, y, z) = 1;
  }
}

void GridSmokeInjector2::injectCircle(
    const ponos::point2f &center, float radius,
    hermes::cuda::GridTexture2<float> &field) {
  auto td = hermes::ThreadArrayDistributionInfo(field.texture().width(),
                                                field.texture().height());
  __injectCircle<<<td.gridSize, td.blockSize>>>(
      field.texture().deviceData(), field.toWorldTransform(),
      hermes::cuda::point2f(center.x, center.y), radius * radius,
      field.texture().width(), field.texture().height());
}

void GridSmokeInjector3::injectSphere(const ponos::point3f &center,
                                      float radius,
                                      hermes::cuda::RegularGrid3Df &field) {
  auto td = hermes::ThreadArrayDistributionInfo(field.resolution());
  __injectSphere<<<td.gridSize, td.blockSize>>>(
      field.accessor(), hermes::cuda::point3f(center.x, center.y, center.z),
      radius * radius);
}

} // namespace cuda

} // namespace poseidon