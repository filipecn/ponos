#include <circe/circe.h>

class HelloCirce : public circe::gl::BaseApp {
public:
  struct Model {
    circe::gl::Program program;
    circe::gl::SceneModel mesh;
    circe::Material material;
    ponos::Transform transform;
  };

  HelloCirce() : BaseApp(800, 800) {
    // setup object model
    ponos::Path assets_path(std::string(ASSETS_PATH));
    loadObject(assets_path + "suzanne.obj");
    object.material.diffuse = object.material.specular = object.material.ambient = circe::Color::White();
    object.material.shininess = 300;
    // setup reference grid
    auto raw_mesh = ponos::RawMeshes::plane(ponos::Plane::XZ(), ponos::point3(), ponos::vec3(100, 0, 0), 20);
    ponos::Path shaders_path(std::string(SHADERS_PATH));
    setupModel(cartesian_plane,
               *raw_mesh,
               {shaders_path + "reference_grid.vert", shaders_path + "reference_grid.frag"});
    // setup light
    light.diffuse = light.specular = light.ambient = circe::Color(0.5, 0.5, 0.5);
    // init shadow map
    shadow_map.setLight(light);
  }

  void prepareFrame(const circe::gl::ViewportDisplay &display) override {
    circe::gl::BaseApp::prepareFrame(display);
    shadow_map.render([&]() {
      object.mesh.draw();
      cartesian_plane.mesh.draw();
    });
  }

  void render(circe::CameraInterface *camera) override {
    drawModel(object, camera);
//    drawModel(cartesian_plane, camera);
    ImGui::Begin("Shadow Map");
    texture_id = shadow_map.depthMap().textureObjectId();
    ImGui::Image((void *) (intptr_t) (texture_id), {1024, 1024});
    ImGui::End();
  }

  void drawModel(Model &model, circe::CameraInterface *camera) const {
    model.program.use();
    model.program.setUniform("light.position", light.point);
    model.program.setUniform("light.ambient", light.ambient.rgb());
    model.program.setUniform("light.diffuse", light.diffuse.rgb());
    model.program.setUniform("light.specular", light.specular.rgb());
    model.program.setUniform("material.kAmbient", model.material.ambient.rgb());
    model.program.setUniform("material.kDiffuse", model.material.diffuse.rgb());
    model.program.setUniform("material.kSpecular", model.material.specular.rgb());
    model.program.setUniform("material.shininess", model.material.shininess);
    model.program.setUniform("model", ponos::transpose(model.transform.matrix()));
    model.program.setUniform("view",
                             ponos::transpose(camera->getViewTransform().matrix()));
    model.program.setUniform("projection",
                             ponos::transpose(camera->getProjectionTransform().matrix()));
    model.program.setUniform("cameraPosition", camera->getPosition());
    model.mesh.draw();
  }

  void loadObject(const ponos::Path &path) {
    ponos::RawMesh rm;
    circe::loadOBJ(path.fullName(), &rm);
    rm.splitIndexData();
    rm.buildInterleavedData();
    ponos::Path shaders_path(std::string(SHADERS_PATH));
    setupModel(object, rm, {shaders_path + "basic.vert", shaders_path + "basic.frag"});
  }

  static void setupModel(Model &model,
                         const ponos::RawMesh &mesh,
                         const std::vector<ponos::Path> &shader_files) {
    model.program.destroy();
    if (!model.program.link(shader_files))
      exit(-1);
    model.mesh.set(mesh);
    model.mesh.bind();
    model.mesh.vertexBuffer().locateAttributes(model.program);
  }

  circe::gl::ShadowMap shadow_map;
  Model object, cartesian_plane;
  circe::Light light;
  GLuint texture_id{0};
};

int main() {
  spdlog::warn("Easy padding in numbers like {:08d}", 12);
  return HelloCirce().run();
}
