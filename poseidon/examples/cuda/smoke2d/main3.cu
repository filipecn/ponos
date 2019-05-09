#include "render.h"

#define WIDTH 800
#define HEIGHT 800

int main() {
  // sim
  ponos::uivec3 resolution(128, 128, 128);
  hermes::cuda::Texture3<float> t(resolution.x, resolution.y, resolution.z);
  hermes::cuda::fill<float>(t, .1f);
  // app
  circe::SceneApp<> app(WIDTH, HEIGHT);
  // cuda interop
  circe::VolumeBox cube(resolution.x, resolution.y, resolution.z);
  CudaGLTextureInterop<float> cgl(cube.texture());
  hermes::cuda::copyPitchedToLinear<float>(t.pitchedData(), cgl.bufferPointer(),
                                           cudaMemcpyDeviceToDevice,
                                           resolution.z);
  // scene
  circe::CartesianGrid grid(5);
  app.scene.add(&grid);
  app.scene.add(&cube);
  app.run();
  return 0;
}