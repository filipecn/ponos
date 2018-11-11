#include <circe/circe.h>
#include <iostream>
#include <ponos/ponos.h>
#include <vector>

circe::SceneApp<> app(800, 800, "Scene App");

int main() {
  WIN32CONSOLE();
  ponos::RawMesh m;
  circe::loadOBJ("C:/Users/fuiri/Desktop/cube.obj", &m);
  m.apply(ponos::scale(5.f, 10.f, 5.f));
  app.init();
  circe::SceneObjectSPtr cg(app.scene.add(new circe::CartesianGrid(5, 5, 5)));
  app.scene.add(new circe::SceneMeshObject(&m, [](circe::ShaderProgram *s) {
    UNUSED_VARIABLE(s);
    glColor4f(0, 0, 0, 0.5);
  }));
  app.run();
  return 0;
}
