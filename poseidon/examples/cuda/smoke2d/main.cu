#include "render.h"
#include <circe/circe.h>
#include <poseidon/poseidon.h>

#define WIDTH 128
#define HEIGHT 128

__global__ void __applyForce(float *f, hermes::cuda::Grid2Info fInfo,
                             hermes::cuda::point2f p, float r, float v) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * fInfo.resolution.x + x;
  if (x < fInfo.resolution.x && y < fInfo.resolution.y) {
    auto wp = fInfo.toWorld(hermes::cuda::point2f(x, y));
    if (hermes::cuda::distance(p, wp) <= r)
      f[index] += v;
  }
}

void applyForce(hermes::cuda::StaggeredGridTexture2 &forceField,
                hermes::cuda::point2f p, float radius, hermes::cuda::vec2f v) {
  {
    auto info = forceField.u().info();
    hermes::ThreadArrayDistributionInfo td(info.resolution.x,
                                           info.resolution.y);
    __applyForce<<<td.gridSize, td.blockSize>>>(forceField.uDeviceData(), info,
                                                p, radius, v.x);
    forceField.u().texture().updateTextureMemory();
  }
  {
    auto info = forceField.v().info();
    hermes::ThreadArrayDistributionInfo td(info.resolution.x,
                                           info.resolution.y);
    __applyForce<<<td.gridSize, td.blockSize>>>(forceField.vDeviceData(), info,
                                                p, radius, v.y);
    forceField.v().texture().updateTextureMemory();
  }
}

int main() {
  // SIM
  poseidon::cuda::GridSmokeSolver2 solver;
  solver.setUIntegrator(new poseidon::cuda::MacCormackIntegrator2());
  solver.setVIntegrator(new poseidon::cuda::MacCormackIntegrator2());
  solver.setIntegrator(new poseidon::cuda::MacCormackIntegrator2());
  solver.setResolution(ponos::uivec2(WIDTH, HEIGHT));
  solver.setDx(1.0 / WIDTH);
  solver.init();
  poseidon::cuda::GridSmokeInjector2::injectCircle(ponos::point2f(0.5f, 0.2f),
                                                   .1f, solver.densityData());
  solver.densityData().texture().updateTextureMemory();
  applyForce(solver.forceFieldData(), hermes::cuda::point2f(0.5, 0.2), 0.1,
             hermes::cuda::vec2f(0, -100));
  // VIS
  circe::SceneApp<> app(WIDTH, HEIGHT, "", false);
  app.addViewport2D(0, 0, WIDTH, HEIGHT);
  CudaOpenGLInterop cgl(WIDTH, HEIGHT);
  circe::ScreenQuad screen;
  screen.shader->begin();
  screen.shader->setUniform("tex", 0);
  app.renderCallback = [&]() {
    solver.step(0.001);
    solver.densityData().texture().updateTextureMemory();
    renderDensity(WIDTH, HEIGHT, solver.densityData().texture(),
                  cgl.bufferPointer());
    // renderScalarGradient(
    //     WIDTH, HEIGHT, solver.pressureData().texture(), cgl.bufferPointer(),
    //     0, 70000, hermes::cuda::Color::blue(), hermes::cuda::Color::red());
    renderSolids(WIDTH, HEIGHT, solver.solidData().texture(),
                 cgl.bufferPointer());
    cgl.sendToTexture();
    cgl.bindTexture(GL_TEXTURE0);
    screen.render();
  };
  app.keyCallback = [&](int key, int scancode, int action, int modifiers) {
    if (action == GLFW_RELEASE) {
      if (key == GLFW_KEY_Q)
        app.exit();
      if (key == GLFW_KEY_SPACE)
        solver.step(0.001);
    }
  };
  app.run();
  return 0;
}