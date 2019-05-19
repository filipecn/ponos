#include "render.h"
#include <poseidon/poseidon.h>

#define WIDTH 800
#define HEIGHT 800

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
  int resSize = 256;
  if (argc > 1)
    sscanf(argv[1], "%d", &resSize);
  // sim
  ponos::uivec3 resolution(resSize);
  hermes::cuda::vec3u res(resSize);
  poseidon::cuda::GridSmokeSolver3 solver;
  solver.addScalarField(); // 0 density
  solver.addScalarField(); // 1 temperature
  solver.setSpacing(ponos::vec3f(1.f / res.x, 1.f / res.z, 1.f / res.z));
  solver.setResolution(resolution);
  // poseidon::cuda::applyEnrightDeformationField(solver.velocity());
  hermes::cuda::fill3(solver.scalarField(0).data().accessor(), 0.f);
  hermes::cuda::fill3(solver.scalarField(1).data().accessor(), 273.f);
  solver.scene.target_temperature.resize(res);
  solver.scene.target_temperature.setSpacing(
      hermes::cuda::vec3f(1.f / res.x, 1.f / res.z, 1.f / res.z));
  solver.scene.smoke_source.resize(res);
  solver.scene.smoke_source.setSpacing(
      hermes::cuda::vec3f(1.f / res.x, 1.f / res.z, 1.f / res.z));
  hermes::cuda::fill3(solver.velocity().u().data().accessor(), 0.f);
  hermes::cuda::fill3(solver.velocity().v().data().accessor(), 0.f);
  hermes::cuda::fill3(solver.velocity().w().data().accessor(), 0.f);
  hermes::cuda::fill3(solver.scene.target_temperature.data().accessor(), 273.f);
  hermes::cuda::fill3(solver.scene.target_temperature,
                      hermes::cuda::bbox3f(hermes::cuda::point3f(0.f),
                                           hermes::cuda::point3f(1.f)),
                      300.f, true);
  hermes::cuda::fill3(solver.scene.smoke_source.data().accessor(),
                      (unsigned char)0);
  hermes::cuda::fill3(
      solver.scene.smoke_source,
      hermes::cuda::bbox3f(hermes::cuda::point3f(0.2f, 0.1f, 0.2f),
                           hermes::cuda::point3f(0.6f, 0.5f, 0.6f)),
      (unsigned char)1, true);
  // std::cerr << solver.scene.target_temperature.data() << std::endl;
  // std::cerr << solver.scene.smoke_source.data() << std::endl;
  solver.init();
  solver.rasterColliders();
  // poseidon::cuda::GridSmokeInjector3::injectSphere(ponos::point3f(0.35f),
  // .15f, solver.scalarField(0));
  // app
  circe::SceneApp<> app(WIDTH, HEIGHT);
  // scene
  hermes::cuda::RegularGrid3Hf solidDensity(res);
  { // solid data for rendering
    hermes::cuda::RegularGrid3Df solidData(res);
    hermes::ThreadArrayDistributionInfo td(res);
    __solidDensity<<<td.gridSize, td.blockSize>>>(solidData.accessor(),
                                                  solver.solid().accessor());
    hermes::cuda::memcpy(solidDensity.data(), solidData.data());
  }
  circe::VolumeBox solid(res.x, res.y, res.z, solidDensity.data().ptr());
  solid.absortion = 4.f;
  circe::VolumeBox cube(res.x, res.y, res.z);
  cube.absortion = 4.f;
  // solver.step(0.01);
  // solver.step(0.01);
  // return;
  // cuda interop
  // CudaGLTextureInterop<float> cgl(cube.texture());
  // cgl.copyFrom(solver.scalarField(0).data().pitchedData());
  // cgl.sendToTexture();
  // cube.update(volumeData.data().ptr());

  hermes::cuda::RegularGrid3Hf volumeData(res);
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
  app.scene.add(&grid);
  app.scene.add(&cube);
  app.scene.add(&solid);
  app.run();
  return 0;
}