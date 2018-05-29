// Created by filipecn on 3/31/18.
#include <aergia/aergia.h>

int main() {
  aergia::SceneApp<> app(800, 800);
  app.init();
//  aergia::TextRenderer textRenderer;
  aergia::FontAtlas font;
  font.loadFont("/mnt/windows/Projects/ponos/aergia/examples/assets/arial.ttf");
  font.setText("asd");
  app.viewports[0].renderCallback = [&]() {
    font.render();
//    textRenderer.withScale(2.f) << textRenderer.withColor(aergia::COLOR_RED)
//                                << textRenderer.at(ponos::Point3()) << "test";
  };
  app.scene.add(new aergia::CartesianGrid(1));
  app.run();
  return 0;
}

