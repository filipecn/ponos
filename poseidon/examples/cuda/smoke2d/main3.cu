#include "render.h"
#include <poseidon/poseidon.h>

#define WIDTH 800
#define HEIGHT 800

int main() {
  // sim
  ponos::uivec3 resolution(256);
  hermes::cuda::vec3u res(64);
  // poseidon::cuda::GridSmokeSolver3<hermes::cuda::StaggeredGridTexture3>
  // solver; solver.setUIntegrator(new poseidon::cuda::MacCormackIntegrator3());
  // solver.setVIntegrator(new poseidon::cuda::MacCormackIntegrator3());
  // solver.setWIntegrator(new poseidon::cuda::MacCormackIntegrator3());
  // solver.setIntegrator(new poseidon::cuda::MacCormackIntegrator3());
  // solver.addScalarField();
  // solver.setResolution(resolution);
  // solver.setDx(1.f / resolution.x);
  // solver.init();
  // poseidon::cuda::GridSmokeInjector3::injectSphere(ponos::point3f(0.5f), .1f,
  //                                                  solver.scalarField(0));
  // hermes::cuda::GridTexture3<float> g, g2, g3, g4;
  // g.resize(hermes::cuda::vec3u(resolution.x, resolution.y, resolution.z));
  // g2.resize(hermes::cuda::vec3u(resolution.x, resolution.y, resolution.z));
  // g3.resize(hermes::cuda::vec3u(resolution.x, resolution.y, resolution.z));
  // g4.resize(hermes::cuda::vec3u(resolution.x, resolution.y, resolution.z));
  // g.setDx(1.f / resolution.x);
  hermes::cuda::RegularGrid3Df g(res);
  g.setSpacing(hermes::cuda::vec3f(1.f / res.x, 1.f / res.z, 1.f / res.z));
  poseidon::cuda::GridSmokeInjector3::injectSphere(ponos::point3f(0.5f), .1f,
                                                   g);
  // app
  circe::SceneApp<> app(WIDTH, HEIGHT);
  // cuda interop
  // std::vector<float> volumeData(resolution.x * resolution.y * resolution.z);
  hermes::cuda::RegularGrid3Hf volumeData(res);
  hermes::cuda::memcpy(volumeData.data(), g.data());
  // hermes::cuda::copyPitchedToLinear(
  //     solver.scalarField(0).texture().pitchedData(), volumeData.data(),
  //     cudaMemcpyDeviceToHost, resolution.z);
  // hermes::cuda::copyPitchedToLinear(g.texture().pitchedData(),
  //                                   volumeData.data(),
  //                                   cudaMemcpyDeviceToHost, resolution.z);
  circe::VolumeBox cube(res.x, res.y, res.z, volumeData.data().ptr());
  // CudaGLTextureInterop<float> cgl(cube.texture());
  // cgl.copyFrom(solver.scalarField(0).texture().pitchedData());
  // cgl.sendToTexture();
  // scene
  circe::CartesianGrid grid(5);
  app.scene.add(&grid);
  app.scene.add(&cube);
  app.run();
  return 0;
}