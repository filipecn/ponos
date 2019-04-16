#include "render.h"
#include <circe/circe.h>
#include <poseidon/poseidon.h>

#define WIDTH 64
#define HEIGHT 64

__global__ void __freeScene(poseidon::cuda::Collider2<float> **solids) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    for (int i = 0; i < 5; ++i)
      delete solids[i];
}

__global__ void __setupScene(poseidon::cuda::Collider2<float> **solids,
                             poseidon::cuda::Collider2<float> **scene) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    solids[0] = new poseidon::cuda::SphereCollider2<float>(
        hermes::cuda::point2(0.f, 0.f), 0.1f);
    float d = 1.0 / 64;
    // floor
    solids[1] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 0.f), hermes::cuda::point2(1.f, d)));
    // ceil
    solids[2] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 1.f - d), hermes::cuda::point2(1.f, 1.f)));
    // left
    solids[3] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 0.f), hermes::cuda::point2(d, 1.f)));
    // right
    solids[4] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(1.f - d, 0.f), hermes::cuda::point2(1.f, 1.f)));
    *scene = new poseidon::cuda::Collider2Set<float>(solids, 5);
  }
}

__global__ void
__rasterColliders(poseidon::cuda::Collider2<float> *const *colliders,
                  unsigned char *solids, const float *u, const float *v,
                  hermes::cuda::Grid2Info sInfo) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * sInfo.resolution.x + x;
  if (x < sInfo.resolution.x && y < sInfo.resolution.y) {
    if ((*colliders)->intersect(sInfo.toWorld(hermes::cuda::point2(x, y))))
      solids[index] = 1;
    else
      solids[index] = 0;
  }
}

int main() {
  // scene
  poseidon::cuda::Scene2<float> scene;
  {
    using namespace hermes::cuda;
    CUDA_CHECK(cudaMalloc(&scene.list,
                          5 * sizeof(poseidon::cuda::Collider2<float> *)));
    CUDA_CHECK(cudaMalloc(&scene.colliders,
                          sizeof(poseidon::cuda::Collider2<float> *)));
  }
  __setupScene<<<1, 1>>>(scene.list, scene.colliders);
  // SIM
  poseidon::cuda::GridSmokeSolver2 solver;
  solver.setResolution(ponos::uivec2(64, 64));
  solver.setDx(1.0 / 64);
  solver.init();
  // solver.rasterColliders(scene);

  {
    hermes::ThreadArrayDistributionInfo td(64, 64);
    __rasterColliders<<<td.gridSize, td.blockSize>>>(
        scene.colliders, solver.solidData().texture().deviceData(),
        solver.solidVelocityData().uDeviceData(),
        solver.solidVelocityData().vDeviceData(), solver.solidData().info());
    solver.solidData().texture().updateTextureMemory();
  }
  std::cerr << solver.solidData().texture() << std::endl;
  poseidon::cuda::GridSmokeInjector2::injectCircle(ponos::point2f(0.2f), .1f,
                                                   solver.densityData());
  solver.densityData().texture().updateTextureMemory();
  hermes::cuda::fill(solver.velocityData().u().texture(), 0.0f);
  hermes::cuda::fill(solver.velocityData().v().texture(), -1.0f);
  // VIS
  circe::SceneApp<> app(WIDTH, HEIGHT, "", false);
  app.addViewport2D(0, 0, WIDTH, HEIGHT);
  CudaOpenGLInterop cgl(WIDTH, HEIGHT);
  circe::ScreenQuad screen;
  screen.shader->begin();
  screen.shader->setUniform("tex", 0);
  app.renderCallback = [&]() {
    solver.step(0.01);
    solver.densityData().texture().updateTextureMemory();
    renderDensity(WIDTH, HEIGHT, solver.densityData().texture(),
                  cgl.bufferPointer());
    renderSolids(WIDTH, HEIGHT, solver.solidData().texture(),
                 cgl.bufferPointer());
    cgl.sendToTexture();
    cgl.bindTexture(GL_TEXTURE0);
    screen.render();
  };
  app.run();
  __freeScene<<<1, 1>>>(scene.list);
  {
    using namespace hermes::cuda;
    CUDA_CHECK(cudaFree(scene.list));
    CUDA_CHECK(cudaFree(scene.colliders));
  }
  return 0;
}