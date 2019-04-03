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
    out[y * w + x] = 0;
    auto cp = toWorld(hermes::cuda::point2f(x, y));
    if ((cp - center).length2() <= radius2)
      out[y * w + x] = 1;
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

} // namespace cuda

} // namespace poseidon