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

class DepthBufferView {
public:
  DepthBufferView() {
    ponos::Path shaders_path(std::string(SHADERS_PATH));
    if (!program.link({shaders_path + "depth_buffer.vert", shaders_path + "depth_buffer.frag"}))
      spdlog::error(program.err);
    std::vector<f32> vertices = {
        // x y u v
        0.0, 0.0, 0.0, 0.0, // 0
        1.0, 0.0, 1.0, 0.0, // 1
        1.0, 1.0, 1.0, 1.0, // 2
        0.0, 1.0, 0.0, 1.0, // 3
    };
    std::vector<u32> indices = {
        0, 1, 2, // 0
        0, 2, 3, // 1
    };
    vb.attributes.pushAttribute<ponos::point2>("position");
    vb.attributes.pushAttribute<ponos::vec2>("uv");
    vb = vertices;
    ib.element_count = 2;
    ib = indices;
    vao.bind();
    vb.bindAttributeFormats();
  }
  void draw() {
    vao.bind();
    vb.bind();
    ib.bind();
    program.use();
    program.setUniform("depthMap", 0);
    ib.draw();
  }
  circe::gl::Program program;
  circe::gl::VertexArrayObject vao;
  circe::gl::VertexBuffer vb;
  circe::gl::IndexBuffer ib;
};

class HelloCirce : public circe::gl::BaseApp {
public:
  struct Model {
    circe::gl::Program program;
    circe::Material material;
    ponos::Transform transform;
    // new way
    circe::gl::VertexArrayObject vao;
    circe::gl::VertexBuffer vb;
    circe::gl::IndexBuffer ib;
  };

  struct SceneUniformBufferData {
    Light light;
  } scene_ubo_data;

  HelloCirce() : BaseApp(800, 800) {
    auto c = this->app_->getCamera();
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
    scene_ubo_data.light.position.y = 10.f;
    scene_ubo_data.light.diffuse = scene_ubo_data.light.specular = scene_ubo_data.light.ambient =
        circe::Color(1.0, 0.5, 0.5);
    // init shadow map
    circe::Light l;
    l.type = circe::LightTypes::DIRECTIONAL;
    l.direction = ponos::vec3(-2.0f, 4.0f, -1.f);
    shadow_map.setLight(l);
    // setup scene uniform buffer
    scene_ubo.push(object.program);
    object.program.setUniformBlockBinding("Scene", 0);
    scene_ubo["Scene"] = &scene_ubo_data;
  }

  void prepareFrame(const circe::gl::ViewportDisplay &display) override {
    circe::gl::BaseApp::prepareFrame(display);
    shadow_map.render([&]() {
      object.ib.draw();
//      cartesian_plane.mesh.draw();
    });
  }

  void render(circe::CameraInterface *camera) override {
    drawModel(object, camera);
    shadow_map.depthMap().bind(GL_TEXTURE0);
    depth_buffer_view.draw();
//    drawModel(cartesian_plane, camera);
    ImGui::Begin("Shadow Map");
    texture_id = shadow_map.depthMap().textureObjectId();
    ImGui::Image((void *) (intptr_t) (texture_id), {256, 256});
    ImGui::End();
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
    glEnable(GL_DEPTH_TEST);
    model.vao.bind();
    model.vb.bind();
    model.ib.bind();
    model.ib.draw();

  }

  void loadObject(const ponos::Path &path) {
    ponos::RawMesh rm;
    circe::loadOBJ(path.fullName(), &rm);
    auto sphere = ponos::RawMeshes::icosphere(ponos::point3(), 1.f, 3, true, false);
    rm.splitIndexData();
    rm.buildInterleavedData();
    ponos::Path shaders_path(std::string(SHADERS_PATH));
//    setupModel(object, *sphere.get(), {shaders_path + "basic.vert", shaders_path + "basic.frag"});
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
    // prepare mesh data
    std::vector<float> vertex_data;
    std::vector<u32> index_data;
    circe::gl::setup_buffer_data_from_mesh(mesh, vertex_data, index_data);
    // describe vertex buffer
    model.vb.attributes.pushAttribute<ponos::point3>("position");
    model.vb.attributes.pushAttribute<ponos::vec3>("normal");
//    model.vb.attributes.pushAttribute<ponos::point2>("uv");
    // upload data
    model.vb = vertex_data;
    model.ib = index_data;
    model.ib.element_count = index_data.size() / 3;
    // bind attributes
    model.vao.bind();
    model.vb.bindAttributeFormats();
  }

  DepthBufferView depth_buffer_view;
  circe::gl::ShadowMap shadow_map;
  circe::gl::UniformBuffer scene_ubo;
  Model object, cartesian_plane;
  GLuint texture_id{0};
};

int main() {
  return HelloCirce().run();
}
