// Created by filipecn on 3/31/18.
#include <aergia/aergia.h>

int main() {
  aergia::SceneApp<> app(800, 800, "", false);
  app.addViewport2D(0,0,800,800);
  aergia::TextRenderer textRenderer
    //   ("/mnt/windows/Projects/ponos/aergia/examples/assets/arial.ttf");
      ("C:/Projects/ponos/aergia/examples/assets/arial.ttf");
  app.viewports[0].renderCallback =
      [&](aergia::CameraInterface *camera) {
        textRenderer.setCamera(camera);
        textRenderer.at(ponos::Point3(1,1,1)) << "blas";
        textRenderer.render("bla", 500, 0, 1.f, aergia::COLOR_RED);
        textRenderer.withScale(.02f) << textRenderer.withColor(aergia::COLOR_BLUE)
                                    << textRenderer.at(ponos::Point3(1,0,0))
                                    << "test";
      };
  app.scene.add(new aergia::CartesianGrid(1));
  app.run();
  return 0;
}

