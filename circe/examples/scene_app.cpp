#include <circe/circe.h>
#include <iostream>
#include <ponos/ponos.h>
#include <vector>

int main() {
  circe::SceneApp<> app(800, 800, "Scene App");
  app.init();
  ponos::RawMesh m;
  auto objPath = std::string(ASSETS_PATH) + "/suzanne.obj";
  circe::loadOBJ(objPath.c_str(), &m);
  m.apply(ponos::scale(5.f, 10.f, 5.f));
  m.buildInterleavedData();
  circe::SceneObjectSPtr cg(app.scene.add(new circe::CartesianGrid(5, 5, 5)));
  app.scene.add(new circe::SceneMeshObject(
      &m, [](circe::ShaderProgram *s, const circe::CameraInterface *camera,
             const ponos::Transform &t) {
        UNUSED_VARIABLE(s);
        glColor4f(0, 0, 0, 0.5);
      }));
  app.run();
  return 0;
}
