#include "render.h"
#include <poseidon/poseidon.h>

using namespace hermes::cuda;
using namespace poseidon::cuda;

int main() {
  vec2u res(100);
  circe::SceneApp<> app(800, 800, "", false);
  app.addViewport2D(0, 0, 800, 800);
  app.getCamera<circe::UserCamera2D>(0)->fit(ponos::bbox2::unitBox());
  LevelSet2H ls(res, vec2f(1.f / res.x));
  for (auto e : ls.grid().accessor())
    e.value = (e.worldPosition() - point2f(0.5f)).length() - 0.2f;
  LevelSet2D d_ls(ls);
  LevelSet2D d_ls2(ls);
  LevelSet2ModelD lsm(d_ls);
  LevelSet2ModelD lsm2(d_ls2);
  RegularGrid2Duc solid(res);
  RegularGrid2Df solid_phi(res);
  StaggeredGrid2D velocity(res, point2f(), vec2f(1.f / res.x));
  fill2(solid.data().accessor(), (unsigned char)0);
  fill2(velocity.u().data().accessor(), 0.f);
  fill2(velocity.v().data().accessor(), 1.f);
  fill2(solid_phi.data().accessor(), 0.f);
  SemiLagrangianIntegrator2 integrator;
  app.scene.add(&lsm);
  int frame = 0;
  auto step = [&]() {
    if (frame++ % 2) {
      integrator.advect(velocity, solid, solid_phi, d_ls2.grid(), d_ls.grid(),
                        0.001f);
      d_ls2.copy(d_ls);
    } else {
      integrator.advect(velocity, solid, solid_phi, d_ls.grid(), d_ls2.grid(),
                        0.001f);
      d_ls.copy(d_ls2);
    }
    lsm.update();
  };
  app.renderCallback = [&]() { step(); };
  app.keyCallback = [&](int key, int scancode, int action, int modifiers) {
    if (action == GLFW_RELEASE) {
      if (key == GLFW_KEY_Q)
        app.exit();
      if (key == GLFW_KEY_SPACE) {
        step();
        // std::cerr << d_ls.grid().data() << std::endl;
      }
    }
  };
  app.run();
  return 0;
}