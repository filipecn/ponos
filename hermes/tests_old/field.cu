#include "utils.h"
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;

TEST(Field, fill) {
  FieldTexture2<float> ft(Transform2<float>(), vec2u(10));
  auto &t = ft.texture();
  fillTexture(t);
  std::vector<float> h_data(t.width() * t.height());
  CHECK_CUDA(cudaMemcpy(h_data.data(), t.deviceData(),
                        h_data.size() * sizeof(float), cudaMemcpyDeviceToHost));
  print(h_data.data(), t.width(), t.height());
}