#include <aergia/aergia.h>
#include <iostream>
#include <ponos/ponos.h>
#include <vector>

#define WIDTH 800
#define HEIGHT 800

int main() {
  aergia::SceneApp<> app(WIDTH, HEIGHT, "Viewports Example", false);
  app.init();
  //app.addViewport2D(0 , 0, WIDTH, HEIGHT);
  //app.getCamera<aergia::UserCamera2D>(0)->setPosition(ponos::Point3(0.5,0,0));
  app.addViewport2D(0 , 0, WIDTH / 2, HEIGHT / 2);
  app.getCamera<aergia::UserCamera2D>(0)->setPosition(ponos::Point3(1,0,0));
  app.addViewport(WIDTH / 2 , 0, WIDTH / 2, HEIGHT / 2);
  app.addViewport2D(WIDTH / 2 , HEIGHT / 2, WIDTH / 2, HEIGHT / 2);
  app.getCamera<aergia::UserCamera2D>(2)->setPosition(ponos::Point3(0,1,0));
  app.addViewport2D(0 , HEIGHT / 2, WIDTH / 2, HEIGHT / 2);
  auto grid = app.scene.add(new aergia::CartesianGrid(5));
  UNUSED_VARIABLE(grid);
  app.run();
  return 0;
}