#include <circe/circe.h>

int main() {
  circe::SceneApp<> app(800, 800, "", false);
  app.addViewport2D(0, 0, 800, 800);
  app.scene.add(new circe::Quad());
  app.scene.add(new circe::CartesianGrid(5));
  app.run();
  return 0;
}
