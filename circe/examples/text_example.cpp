// Created by filipecn on 3/31/18.
#include <circe/circe.h>

int main() {
  circe::SceneApp<> app(800, 800, "", false);
  app.addViewport2D(0, 0, 800, 800);
  circe::TextRenderer textRenderer(
      "/mnt/windows/Projects/ponos/circe/examples/assets/arial.ttf");
  // ("C:/Projects/ponos/circe/examples/assets/arial.ttf");
  app.viewports[0].renderCallback = [&](circe::CameraInterface *camera) {
    textRenderer.setCamera(camera);
    textRenderer.at(ponos::point3f(1, 1, 1)) << "blas";
    textRenderer.render("bla", 500, 0, 1.f, circe::COLOR_RED);
    textRenderer.withScale(.02f)
        << textRenderer.withColor(circe::COLOR_BLUE)
        << textRenderer.at(ponos::point3f(1, 0, 0)) << "test";
  };
  app.scene.add(new circe::CartesianGrid(1));
  int font_id = circe::FontManager::loadFromFile(
      "/mnt/windows/Projects/ponos/circe/examples/assets/arial.ttf");
  std::vector<circe::TextObject> list;
  for (int i = 0; i < 100; i++) {
    list.emplace_back(font_id);
    list[i].setText(ponos::concat("t", i));
    list[i].text_color = circe::COLOR_RED;
    list[i].position = ponos::point3f((i % 10) * 0.1, (i / 10) * 0.1, 0.5);
    list[i].text_size = 0.001;
  }
  for (int i = 0; i < 100; i++)
    app.scene.add(&list[i]);
  app.run();
  return 0;
}
