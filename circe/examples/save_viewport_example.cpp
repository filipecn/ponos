// Created by filipecn on 9/3/18.
#include <circe/circe.h>
#include <lodepng.h>

int main() {
  circe::SceneApp<> app(800, 800);
  app.scene.add(new circe::CartesianGrid(5));
  app.buttonCallback = [&](int button, int action, int modifiers) {
    if (action == GLFW_RELEASE) {
      std::vector<unsigned char> data;
      size_t w = 0, h = 0;
      app.viewports[0].renderer->currentPixels(data, w, h);
      unsigned error = lodepng::encode("test.png", data, w, h);
    }
  };
  app.run();
  return 0;
}
