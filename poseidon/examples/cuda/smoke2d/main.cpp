#include "render.h"
#include <circe/circe.h>
#include <poseidon/poseidon.h>

#define WIDTH 64
#define HEIGHT 64

int main() {
  // SIM
  poseidon::cuda::GridSmokeSolver2 solver;
  solver.setResolution(ponos::uivec2(64, 64));
  poseidon::cuda::GridSmokeInjector2::injectCircle(ponos::point2f(50.f), 20.f,
                                                   solver.densityData());
  solver.densityData().texture().updateTextureMemory();
  hermes::cuda::fill(solver.velocityData().u().texture(), 0.0f);
  hermes::cuda::fill(solver.velocityData().v().texture(), 0.0f);
  std::cerr << solver.velocityData().u().texture() << std::endl;
  std::cerr << solver.velocityData().v().texture() << std::endl;
  std::cerr << solver.densityData().texture() << std::endl;
  for (int i = 0; i < 10; i++) {
    solver.step(0.001);
    solver.densityData().texture().updateTextureMemory();
  }
  std::cerr << solver.densityData().texture() << std::endl;
  std::cerr << solver.densityData().toWorldTransform().getMatrix() << std::endl;
  std::cerr << solver.densityData().toWorldTransform()(
      hermes::cuda::point2f(0.5f));
  std::cerr << solver.densityData().toFieldTransform()(
      hermes::cuda::point2f(0.5f));
  std::cerr << solver.velocityData().u().toFieldTransform()(
                   hermes::cuda::point2f(0.5f))
            << std::endl;
  std::cerr << solver.velocityData().v().toFieldTransform()(
                   hermes::cuda::point2f(0.5f))
            << std::endl;
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
    cgl.sendToTexture();
    cgl.bindTexture(GL_TEXTURE0);
    screen.render();
  };
  app.run();
  return 0;
}