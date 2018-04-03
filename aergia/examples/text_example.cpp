// Created by filipecn on 3/31/18.
#include <aergia/aergia.h>

int main() {
  aergia::SceneApp<> app(800, 800);
  app.init();
  aergia::TextRenderer textRenderer;
  app.viewports[0].renderCallback = [&]() {
    textRenderer.withScale(2.f) << textRenderer.withColor(aergia::COLOR_RED)
                                << textRenderer.at(ponos::Point3()) << "test";
  };
  app.scene.add(new aergia::CartesianGrid(1));
  app.run();
  return 0;
}

