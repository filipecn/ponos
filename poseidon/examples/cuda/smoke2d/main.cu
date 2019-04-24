#include "render.h"
#include <circe/circe.h>
#include <poseidon/poseidon.h>

#define WIDTH 128
#define HEIGHT 128

int main() {
  // SIM
  poseidon::cuda::GridSmokeSolver2 solver;
  solver.setResolution(ponos::uivec2(WIDTH, HEIGHT));
  solver.setDx(1.0 / WIDTH);
  solver.init();
  poseidon::cuda::GridSmokeInjector2::injectCircle(ponos::point2f(0.2f), .1f,
                                                   solver.densityData());
  solver.densityData().texture().updateTextureMemory();
  // hermes::cuda::fill(solver.velocityData().v().texture(), -1.0f);
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
    renderSolids(WIDTH, HEIGHT, solver.solidData().texture(),
                 cgl.bufferPointer());
    renderScalarGradient(
        WIDTH, HEIGHT, solver.densityData().texture(), cgl.bufferPointer(), 0,
        1, hermes::cuda::Color::green(), hermes::cuda::Color::red());
    cgl.sendToTexture();
    cgl.bindTexture(GL_TEXTURE0);
    screen.render();
  };
  app.run();
  return 0;
}