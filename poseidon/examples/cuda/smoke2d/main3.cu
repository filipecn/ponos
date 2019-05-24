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

__global__ void __sampleField(hermes::cuda::RegularGrid3Accessor<float> field,
                              hermes::cuda::RegularGrid3Accessor<float> out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (out.isIndexStored(x, y, z)) {
    out(x, y, z) = field(field.worldPosition(x, y, z));
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
  // hermes::cuda::fill3(solver.velocity().w(), hermes::cuda::bbox3f::unitBox(),
  //                     1.f);
  // hermes::cuda::fill3(solver.velocity().v(), hermes::cuda::bbox3f::unitBox(),
  //                     -1.f);
  // hermes::cuda::fill3(solver.velocity().w(), hermes::cuda::bbox3f::unitBox(),
  //                     -1.f);
  auto size = res;
  {
    // hermes::cuda::RegularGrid3Hf h_density(res);
    // auto hsAcc = h_density.accessor();
    // for (int k = 0; k < size.z; k++)
    //   for (int j = 0; j < size.y; j++)
    //     for (int i = 0; i < size.x; i++)
    //       if (i >= 14 && i <= 16 && j >= 14 && j <= 16 && k >= 14 && k <= 16)
    //         hsAcc(i, j, k) = 1.f;
    //       else
    //         hsAcc(i, j, k) = 0.f;
    // memcpy(solver.scalarField(0).data(), h_density.data());
    // std::cerr << solver.scalarField(0).data() << std::endl;
  }
  hermes::cuda::RegularGrid3Huc h_solid(size);
  auto hsAcc = h_solid.accessor();
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++)
        if (i > 0 && i < size.x - 1 && j > 0 && j < size.y - 1 && k > 0 &&
            k < size.z - 1)
          hsAcc(i, j, k) = 0;
        else
          hsAcc(i, j, k) = 1;
  hermes::cuda::memcpy(solver.solid().data(), h_solid.data());

  // hermes::cuda::fill3(
  //     solver.scene().smoke_source,
  //     hermes::cuda::bbox3f(hermes::cuda::point3f(0.2f, 0.1f, 0.2f),
  //                          hermes::cuda::point3f(0.6f, 0.5f, 0.6f)),
  //     (unsigned char)1, true);
  poseidon::cuda::GridSmokeInjector3::injectSphere(ponos::point3f(0.35f), .15f,
                                                   solver.scalarField(0));
  hermes::cuda::RegularGrid3Df sampledData(res);
  { // sample density
    // hermes::ThreadArrayDistributionInfo td(res);
    // __sampleField<<<td.gridSize, td.blockSize>>>(
    //     solver.scalarField(0).accessor(), sampledData.accessor());
  }
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
    // hermes::cuda::memcpy(volumeData.data(), sampledData.data());
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