#include <aergia/aergia.h>

int main() {
  aergia::SceneApp<> app(800, 800, "", false);
  app.addViewport2D(0,0,800,800);
  app.scene.add(new aergia::Quad());
  app.scene.add(new aergia::CartesianGrid(5));
  app.run();
  return 0;
}
