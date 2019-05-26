#include "render.h"
#include <lodepng.h>
#include <poseidon/poseidon.h>

#define WIDTH 800
#define HEIGHT 800

inline void flipImage(std::vector<unsigned char> &data, size_t w, size_t h) {
  for (uint x = 0; x < w; x++)
    for (uint y = 0; y < h / 2; y++)
      for (uint k = 0; k < 4; k++) {
        unsigned char tmp = data[4 * w * (h - 1 - y) + 4 * x + k];
        data[4 * w * (h - 1 - y) + 4 * x + k] = data[4 * w * y + 4 * x + k];
        data[4 * w * y + 4 * x + k] = tmp;
      }
}

__global__ void
__solidDensity(hermes::cuda::RegularGrid3Accessor<float> density,
               hermes::cuda::RegularGrid3Accessor<unsigned char> solid) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (solid.isIndexStored(x, y, z)) {
    density(x, y, z) = (float)solid(x, y, z);
  }
}

int main(int argc, char **argv) {
  int resSize = 128;
  if (argc > 1)
    sscanf(argv[1], "%d", &resSize);
  // sim
  ponos::uivec3 resolution(resSize);
  hermes::cuda::vec3u res(resSize);
  poseidon::cuda::GridSmokeSolver3 solver;
  solver.setSpacing(ponos::vec3f(1.f / res.x, 1.f / res.y, 1.f / res.z));
  solver.setResolution(resolution);
  solver.init();
  // solver.rasterColliders();
  // poseidon::cuda::applyEnrightDeformationField(solver.velocity());
  auto size = res;
  hermes::cuda::RegularGrid3Huc h_solid(size);
  auto hsAcc = h_solid.accessor();
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++)
        if (i > 0 && i < size.x - 1 && j > 0 /*&& j < size.y - 1*/ && k > 0 &&
            k < size.z - 1)
          hsAcc(i, j, k) = 0;
        else
          hsAcc(i, j, k) = 1;
  hermes::cuda::memcpy(solver.solid().data(), h_solid.data());

  // hermes::cuda::fill3(
  //     solver.scene().smoke_source,
  //     hermes::cuda::bbox3f(hermes::cuda::point3f(0.45f, 0.1f, 0.45f),
  //                          hermes::cuda::point3f(0.55f, 0.3f, 0.55f)),
  //     (unsigned char)1);
  // hermes::cuda::fill3(
  //     // solver.scene().target_temperature,
  //     solver.scalarField(1),
  //     hermes::cuda::bbox3f(hermes::cuda::point3f(0.4f, 0.1f, 0.4f),
  //                          hermes::cuda::point3f(0.6f, 0.15f, 0.6f)),
  //     350.f);
  poseidon::cuda::GridSmokeInjector3::injectSphere(
      ponos::point3f(0.5f, 0.65f, 0.5f), .08f, solver.scalarField(0));
  // app
  circe::SceneApp<> app(WIDTH, HEIGHT);
  app.getCamera<>(0)->setPosition(ponos::point3f(2, 0.7, 2));
  // scene
  // hermes::cuda::RegularGrid3Hf solidDensity(res);
  // { // solid data for rendering
  //   hermes::cuda::RegularGrid3Df solidData(res);
  //   hermes::ThreadArrayDistributionInfo td(res);
  //   __solidDensity<<<td.gridSize, td.blockSize>>>(solidData.accessor(),
  //                                                 solver.solid().accessor());
  //   hermes::cuda::memcpy(solidDensity.data(), solidData.data());
  // }
  // circe::VolumeBox solid(res.x, res.y, res.z, solidDensity.data().ptr());
  // solid.absortion = 4.f;
  circe::VolumeBox cube(res.x, res.y, res.z);
  cube.absortion = 10.f;
  cube.lightIntensity = ponos::vec3f(20.f, 20.f, 20.f);
  cube.lightPos = ponos::vec3f(2, 0.7, 2);
  // cuda interop
  // CudaGLTextureInterop<float> cgl(cube.texture());
  // cgl.copyFrom(solver.scalarField(0).data().pitchedData());
  // cgl.sendToTexture();
  // cube.update(volumeData.data().ptr());
  hermes::cuda::RegularGrid3Hf volumeData(res);
  int frame = 0;
  app.viewports[0].renderEndCallback = [&]() {
    std::vector<unsigned char> image;
    size_t w, h;
    app.viewports[0].renderer->currentPixels(image, w, h);
    flipImage(image, w, h);
    char filename[20];
    sprintf(filename, "%d.png", frame++);
    unsigned error =
        lodepng::encode(filename, &image[0], static_cast<unsigned int>(w),
                        static_cast<unsigned int>(h));
    if (error)
      std::cout << "encoder error " << error << ": "
                << lodepng_error_text(error) << std::endl;
  };
  app.renderCallback = [&]() {
    solver.step(0.01);
    hermes::cuda::memcpy(volumeData.data(), solver.scalarField(0).data());
    cube.update(volumeData.data().ptr());
  };
  app.keyCallback = [&](int key, int scancode, int action, int modifiers) {
    if (action == GLFW_RELEASE) {
      if (key == GLFW_KEY_Q)
        app.exit();
      if (key == GLFW_KEY_SPACE)
        solver.step(0.01);
    }
  };
  circe::CartesianGrid grid(5);
  // app.scene.add(&grid);
  app.scene.add(&cube);
  // app.scene.add(&solid);
  app.run();
  return 0;
}