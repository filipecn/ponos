// Created by filipecn on 3/31/18.
#include <circe/circe.h>

int main() {
  circe::SceneApp<> app(800, 800, "", false);
  app.addViewport2D(0, 0, 800, 800);
  circe::TextRenderer textRenderer
      //   ("/mnt/windows/Projects/ponos/circe/examples/assets/arial.ttf");
      ("C:/Projects/ponos/circe/examples/assets/arial.ttf");
  app.viewports[0].renderCallback = [&](circe::CameraInterface *camera) {
    textRenderer.setCamera(camera);
    textRenderer.at(ponos::Point3(1, 1, 1)) << "blas";
    textRenderer.render("bla", 500, 0, 1.f, circe::COLOR_RED);
    textRenderer.withScale(.02f) << textRenderer.withColor(circe::COLOR_BLUE)
                                 << textRenderer.at(ponos::Point3(1, 0, 0))
                                 << "test";
  };
  app.scene.add(new circe::CartesianGrid(1));
  app.run();
  return 0;
}
