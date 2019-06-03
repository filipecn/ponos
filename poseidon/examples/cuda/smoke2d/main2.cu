#include "render.h"
#include <lodepng.h>
#include <poseidon/poseidon.h>

using namespace hermes::cuda;
using namespace poseidon::cuda;

#define WIDTH 800
#define HEIGHT 800

int main(int argc, char **argv) {
  int resSize = 128;
  if (argc > 1)
    sscanf(argv[1], "%d", &resSize);
  // sim
  ponos::uivec2 resolution(resSize);
  vec2u res(resSize);
  GridSmokeSolver2 solver;
  solver.setSpacing(ponos::vec2f(1.f / res.x, 1.f / res.y));
  solver.setResolution(resolution);
  solver.init();
  auto size = res;
  hermes::cuda::RegularGrid2Huc h_solid(size);
  for (auto e : h_solid.accessor())
    // if (e.i() > 3 && e.i() < size.x - 4 && e.j() > 3)
    e.value = 0;
  // else
  //   e.value = 1;
  hermes::cuda::memcpy(solver.solid().data(), h_solid.data());
  applyZalesakDeformationField(solver.velocity());

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
  poseidon::cuda::GridSmokeInjector2::injectCircle(ponos::point2f(0.5f, 0.65f),
                                                   .05f, solver.scalarField(0));
  // app
  circe::SceneApp<> app(WIDTH, HEIGHT, "", false);
  app.addViewport2D(0, 0, WIDTH, HEIGHT);
  app.getCamera<circe::UserCamera2D>(0)->fit(ponos::bbox2::unitBox());
  // cuda interop
  CudaOpenGLInterop cgl(res.x, res.y);
  circe::ScreenQuad screen;
  screen.shader->begin();
  screen.shader->setUniform("tex", 0);
  app.renderCallback = [&]() {
    solver.step(0.01);
    renderDensity(solver.scalarField(0), cgl.bufferPointer<unsigned int>());
    cgl.sendToTexture();
    cgl.bindTexture(GL_TEXTURE0);
    screen.render();
  };
  app.keyCallback = [&](int key, int scancode, int action, int modifiers) {
    if (action == GLFW_RELEASE) {
      if (key == GLFW_KEY_Q)
        app.exit();
      if (key == GLFW_KEY_SPACE) {
        solver.step(0.01);
        // std::cerr << solver.scalarField(0).data() << std::endl;
      }
    }
  };
  circe::CartesianGrid grid(5);
  // app.scene.add(&grid);
  app.run();
  return 0;
}