#include "render.h"
#include <lodepng.h>
#include <poseidon/poseidon.h>

using namespace hermes::cuda;
using namespace poseidon::cuda;

inline void flipImage(std::vector<unsigned char> &data, size_t w, size_t h) {
  for (uint x = 0; x < w; x++)
    for (uint y = 0; y < h / 2; y++)
      for (uint k = 0; k < 4; k++) {
        unsigned char tmp = data[4 * w * (h - 1 - y) + 4 * x + k];
        data[4 * w * (h - 1 - y) + 4 * x + k] = data[4 * w * y + 4 * x + k];
        data[4 * w * y + 4 * x + k] = tmp;
      }
}

int main(int argc, char **argv) {
  int r = 128;
  if (argc > 1)
    r = atoi(argv[1]);
  vec3u res(r);
  vec3f s(1.0 / 128.0);
  circe::SceneApp<> app(800, 800, "");
  app.getCamera<>(0)->setPosition(ponos::point3f(1, 0.7, 1));
  PracticalLiquidsSolver3D pls;
  pls.setResolution(res);
  pls.setSpacing(s);
  // walls
  int wall_thick = 3;
  pls.rasterSolid(bbox3f(point3f(-1), point3f(wall_thick * s.x, 2.f, 2.f)),
                  vec3f());
  pls.rasterSolid(bbox3f(point3f(-1), point3f(2.f, wall_thick * s.y, 2.f)),
                  vec3f());
  pls.rasterSolid(bbox3f(point3f(-1), point3f(2.f, 2.f, wall_thick * s.z)),
                  vec3f());
  pls.rasterSolid(
      bbox3f(point3f(res.x * s.x - wall_thick * s.x, -1, -1), point3f(2)),
      vec3f());
  // pls.rasterSolid(bbox3f(point3f(-1, 1.f - wall_thick * s.y, -1),
  // point3f(2)),
  //                 vec3f());
  pls.rasterSolid(
      bbox3f(point3f(-1, -1, res.x * s.x - wall_thick * s.z), point3f(2)),
      vec3f());
  LevelSet3H ls(pls.surfaceLevelSet());
  for (auto e : ls.grid().accessor()) {
    e.value = 1000.f;
    // e.value =
    //     min(e.value, (e.worldPosition() - point3f(0.35f)).length() - .15f);
    e.value = min(e.value, (e.worldPosition() - point3f(res.x * 0.5 * s.x,
                                                        res.y * 0.8 * s.y,
                                                        res.z * 0.5 * s.z))
                                   .length() -
                               4 * s.x);
    e.value = min(e.value, e.worldPosition().y - res.x * 0.4 * s.x);
  }
  // poseidon::cuda::applyEnrightDeformationField(pls.velocity());
  pls.surfaceLevelSet().copy(ls);
  auto force = pls.forceField();
  fill3(force.v().data(), -9.81f);
  fill3(pls.velocity().v().data(), -0.5f);
  for (int i = 0; i < 1; i++)
    pls.step(0.001f);
  // return 0;
  LevelSet3ModelD lsm(pls.surfaceLevelSet());
  LevelSet3ModelD slsm(pls.solidLevelSet());
  LevelSet3DistancesModelD lsdm(pls.surfaceLevelSet());
  // slsm.color = circe::Color::Yellow();
  app.scene.add(&lsm);
  // app.scene.add(&lsdm);
  // app.scene.add(&slsm);
  slsm.update();
  lsdm.update();
  app.renderCallback = [&]() {
    pls.step(1.f / 60.f);
    lsm.update();
  };
  int frame = 0;
  app.viewports[0].renderEndCallback = [&]() {
    // return;
    if (frame > 1000)
      exit(0);
    std::vector<unsigned char> image;
    size_t w, h;
    app.viewports[0].renderer->currentPixels(image, w, h);
    flipImage(image, w, h);
    char filename[20];
    sprintf(filename, "%d.png", frame++);
    unsigned error =
        lodepng::encode(filename, &image[0], static_cast<unsigned int>(w),
                        static_cast<unsigned int>(h));
    if (error)
      std::cout << "encoder error " << error << ": "
                << lodepng_error_text(error) << std::endl;
  };
  app.keyCallback = [&](int key, int scancode, int action, int modifiers) {
    if (action == GLFW_PRESS) {
      if (key == GLFW_KEY_Q)
        app.exit();
      if (key == GLFW_KEY_D)
        lsdm.visible = !lsdm.visible;
      if (key == GLFW_KEY_SPACE) {
        pls.step(1.f / 60.f);
        lsm.update();
        lsdm.update();
      }
    }
  };
  circe::CartesianGrid grid(res.x);
  // grid.setDimension(0, 0, res.x);
  // grid.setDimension(1, 0, res.y);
  // grid.setDimension(2, 0, res.z);
  grid.transform = ponos::scale(s.x, s.y, s.z);
  // app.scene.add(&grid);
  app.run();
  return 0;
}