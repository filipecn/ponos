#include <circe/circe.h>

struct alignas(16) vec3_16 {
  vec3_16 &operator=(const circe::Color &color) {
    x = color.r;
    y = color.g;
    z = color.b;
    return *this;
  }
  float x{0};
  float y{0};
  float z{0};
};

struct Light {
  vec3_16 position;
  vec3_16 ambient;
  vec3_16 diffuse;
  vec3_16 specular;
};

class HelloCirce : public circe::gl::BaseApp {
public:
  struct Model {
    circe::gl::Program program;
    circe::gl::SceneModel mesh;
    circe::Material material;
    ponos::Transform transform;
  };

  struct SceneUniformBufferData {
    Light light;
  } scene_ubo_data;

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
    scene_ubo_data.light.position.x = 10.f;
    scene_ubo_data.light.diffuse = scene_ubo_data.light.specular = scene_ubo_data.light.ambient =
        circe::Color(0.5, 0.5, 0.5);
    // init shadow map
//    shadow_map.setLight(scene_ubo_data.light);
// setup scene uniform buffer
    scene_ubo.push(object.program);
    object.program.setUniformBlockBinding("Scene", 0);
    scene_ubo["Scene"] = &scene_ubo_data;
  }

  void prepareFrame(const circe::gl::ViewportDisplay &display) override {
    circe::gl::BaseApp::prepareFrame(display);
//    shadow_map.render([&]() {
//      object.mesh.draw();
//      cartesian_plane.mesh.draw();
//    });
  }

  void render(circe::CameraInterface *camera) override {
    drawModel(object, camera);
//    drawModel(cartesian_plane, camera);
//    ImGui::Begin("Shadow Map");
//    texture_id = shadow_map.depthMap().textureObjectId();
//    ImGui::Image((void *) (intptr_t) (texture_id), {1024, 1024});
//    ImGui::End();
  }

  void drawModel(Model &model, circe::CameraInterface *camera) {
    model.program.use();
//    model.program.setUniform("light.position", scene_ubo_data.light.position);
//    model.program.setUniform("light.ambient", scene_ubo_data.light.ambient);
//    model.program.setUniform("light.diffuse", scene_ubo_data.light.diffuse);
//    model.program.setUniform("light.specular", scene_ubo_data.light.specular);
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
    if (!model.program.link(shader_files)) {
      spdlog::error(model.program.err);
      exit(-1);
    }
    model.mesh.set(mesh);
    model.mesh.bind();
    model.mesh.vertexBuffer().locateAttributes(model.program);
  }

//  circe::gl::ShadowMap shadow_map;
  circe::gl::UniformBuffer scene_ubo;
  Model object, cartesian_plane;
  GLuint texture_id{0};
};

int main() {
  return HelloCirce().run();
  circe::gl::SceneApp<> app(800, 800);

  std::vector<f32> vertices = {
      // x   y    z    u    v
      0.f, 0.f, 0.f, 0.f, 0.f, // 0
      1.f, 0.f, 0.f, 1.f, 0.f, // 1
      1.f, 1.f, 0.f, 1.f, 1.f, // 2
      0.f, 0.f, 1.f, 0.f, 1.f,  // 3
      0.f, 1.f, 1.f, 1.f, 1.f, // 4
  };
  std::vector<u32> indices = {
      0, 1, 2, // 0
      3, 4, 0, // 1
  };

//  circe::gl::DeviceMemory dmv(GL_STATIC_DRAW, GL_ARRAY_BUFFER,
//                              vertices.size() * sizeof(f32),
//                              vertices.data());
//  circe::gl::DeviceMemory dmi(GL_STATIC_DRAW, GL_ELEMENT_ARRAY_BUFFER,
//                              indices.size() * sizeof(i32),
//                              indices.data());
  circe::gl::VertexBuffer vb;
  vb.attributes.pushAttribute<ponos::point3>("position");
  vb.attributes.pushAttribute<ponos::point2>("uv");
  vb = vertices;
  circe::gl::IndexBuffer ib;
  ib = indices;

  ponos::Path shaders_path(std::string(SHADERS_PATH));
  circe::gl::Program p({shaders_path + "basic.vert",
                        shaders_path + "basic.frag"});
  std::cerr << p << std::endl;

  circe::gl::UniformBuffer ub;
  ub.push(p);
  std::cerr << ub << std::endl;
//  ub["Transforms"] = &transforms;

  return app.run();

}
