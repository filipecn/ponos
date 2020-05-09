#include <circe/circe.h>

int main() {
  circe::gl::SceneApp<> app(800, 800, "");
  app.scene.add(new circe::gl::CartesianGrid(5));
  auto objPath = std::string(ASSETS_PATH) + "/suzanne.obj";
  //   objPath = "C:/Users/fuiri/Desktop/dragon.obj";
  auto vs = std::string(SHADERS_PATH) + "/basic.vert";
  auto fs = std::string(SHADERS_PATH) + "/basic.frag";
  auto shader = circe::gl::createShaderProgramPtr(
      circe::gl::ShaderManager::instance().loadFromFiles({vs.c_str(), fs.c_str()}));
  shader->addVertexAttribute("position", 0);
  shader->addVertexAttribute("normal", 1);
  shader->addUniform("Light.position", 2);
  shader->addUniform("Light.ambient", 3);
  shader->addUniform("Light.diffuse", 4);
  shader->addUniform("Light.specular", 5);
  shader->addUniform("Material.kAmbient", 6);
  shader->addUniform("Material.kDiffuse", 7);
  shader->addUniform("Material.kSpecular", 8);
  shader->addUniform("Material.shininess", 9);
  shader->addUniform("cameraPosition", 10);
  shader->addUniform("model", 11);
  shader->addUniform("view", 12);
  shader->addUniform("projection", 13);
  auto obj = circe::gl::createSceneMeshObjectSPtr(objPath, shader);
//  std::cerr << *obj->mesh()->rawMesh() << std::endl;
  obj->transform = ponos::scale(0.5, 0.5, 0.5);
  obj->drawCallback = [](circe::gl::ShaderProgram *s,
                         const circe::CameraInterface *camera,
                         ponos::Transform t) {
    glEnable(GL_DEPTH_TEST);
    s->begin();
    s->setUniform("Light.position", ponos::vec3(0, 0, 5));
    s->setUniform("Light.ambient", ponos::vec3(1, 1, 1));
    s->setUniform("Light.diffuse", ponos::vec3(1, 1, 1));
    s->setUniform("Light.specular", ponos::vec3(1, 1, 1));
    s->setUniform("Material.kAmbient", ponos::vec3(0.01, 0.01, 0.1));
    s->setUniform("Material.kDiffuse", ponos::vec3(0.5, 0.5, 0.5));
    s->setUniform("Material.kSpecular", ponos::vec3(0.8, 1, 1));
    s->setUniform("Material.shininess", 200.f);
    s->setUniform("model", ponos::transpose(t.matrix()));
    s->setUniform("view",
                  ponos::transpose(camera->getViewTransform().matrix()));
    s->setUniform("projection",
                  ponos::transpose(camera->getProjectionTransform().matrix()));
    s->setUniform("cameraPosition", camera->getPosition());
  };
  app.scene.add(obj.get());
  app.run();
  return 0;
}
