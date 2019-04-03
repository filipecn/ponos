#include <hermes/hermes.h>
#include <iostream>

inline void print(const float *data, unsigned int width, unsigned int height) {
  for (unsigned int i = 0; i < width; i++) {
    for (unsigned int j = 0; j < height; j++)
      std::cerr << data[i * width + j] << ",";
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
}

void fillTexture(hermes::cuda::Texture<float> &t);
