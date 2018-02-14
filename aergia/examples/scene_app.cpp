#include <aergia/aergia.h>
#include <iostream>
#include <ponos/ponos.h>
#include <vector>

aergia::SceneApp<> app(800, 800, "Scene App");

int main() {
  WIN32CONSOLE();
  ponos::RawMesh m;
  aergia::loadOBJ("C:/Users/fuiri/Desktop/cube.obj", &m);
  m.apply(ponos::scale(5.f, 10.f, 5.f));
  app.init();
  aergia::SceneObjectSPtr cg(app.scene.add(new aergia::CartesianGrid(5, 5, 5)));
  app.scene.add(new aergia::SceneMesh(&m, [](aergia::Shader *s) {
    UNUSED_VARIABLE(s);
    glColor4f(0, 0, 0, 0.5);
  }));
  app.run();
  return 0;
}
