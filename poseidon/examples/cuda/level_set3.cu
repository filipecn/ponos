#include "render.h"
#include <poseidon/poseidon.h>

using namespace hermes::cuda;
using namespace poseidon::cuda;

int main() {
  vec3u res(100);
  circe::SceneApp<> app(800, 800, "");
  LevelSet3H ls(res, vec3f(1.f / res.x));
  for (auto e : ls.grid().accessor())
    e.value = (e.worldPosition() - point3f(0.5f)).length() - 0.2f;
  LevelSet3D d_ls(ls);
  LevelSet3D d_ls2(ls);
  LevelSet3ModelD lsm(d_ls);
  LevelSet3ModelD lsm2(d_ls2);
  app.scene.add(&lsm);
  circe::CartesianGrid grid(1);
  app.scene.add(&grid);
  app.run();
  return 0;
}