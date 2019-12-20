#include <poseidon/poseidon.h>
#include <circe/circe.h>

using namespace poseidon;
using namespace circe;

int main(int argc, char **argv) {
  // simulation data
  GridSmokeSolver2 solver;
  solver.setResolution(ponos::size2(64, 64));
  solver.setSpacing(ponos::vec2(0.01, 0.01));
  // visualization
  circe::SceneApp<> app(800, 800, "", false);
  app.addViewport2D(0, 0, 800, 800);
  app.getCamera<circe::UserCamera2D>(0)->fit(ponos::bbox2::unitBox());
  CartesianGrid g(5);
  app.scene.add(&g);
  return app.run();
}

