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
    // setup texture image
    circe::gl::TextureAttributes attributes;
    attributes.target = GL_TEXTURE_2D;
    attributes.width = 256;
    attributes.height = 256;
    attributes.type = GL_UNSIGNED_BYTE;
    attributes.internal_format = GL_RGBA8;
    attributes.format = GL_RGBA;
    circe::gl::TextureParameters parameters;
    image.set(attributes, parameters);
    // setup shader
    ponos::Path shaders_path(std::string(SHADERS_PATH));
    if (!program.link({shaders_path + "depth_buffer.vert", shaders_path + "depth_buffer.frag"}))
      spdlog::error(program.err);
    // setup screen quad
    std::vector<f32> vertices = {
        // x y u v
        -1.0, -1.0, 0.0, 0.0, // 0
        1.0, -1.0, 1.0, 0.0, // 1
        1.0, 1.0, 1.0, 1.0, // 2
        -1.0, 1.0, 0.0, 1.0, // 3
    };
    std::vector<u32> indices = {
        0, 1, 2, // 0
        0, 2, 3, // 1
    };
    vb.attributes.push<ponos::point2>("position");
    vb.attributes.push<ponos::vec2>("uv");
    vb = vertices;
    ib.element_count = 2;
    ib = indices;
    vao.bind();
    vb.bindAttributeFormats();
  }
  void update() {
    image.render([&]() {
      vao.bind();
      vb.bind();
      ib.bind();
      program.use();
      program.setUniform("depthMap", 0);
      ib.draw();
    });
  }
  circe::gl::Program program;
  circe::gl::VertexArrayObject vao;
  circe::gl::VertexBuffer vb;
  circe::gl::IndexBuffer ib;
  circe::gl::RenderTexture image;
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
    void draw() {
      vao.bind();
      vb.bind();
      ib.bind();
      ib.draw();
    }
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
    // init shadow map
    circe::Light l;
    l.type = circe::LightTypes::DIRECTIONAL;
    l.direction = ponos::normalize(ponos::vec3(-1.0f, 1.0f, 1.f));
    shadow_map.setLight(l);
    // setup scene uniform buffer
    scene_ubo_data.light.position.x = l.direction.x * 20;
    scene_ubo_data.light.position.y = l.direction.y * 20;
    scene_ubo_data.light.position.z = l.direction.z * 20;
    scene_ubo_data.light.diffuse = scene_ubo_data.light.specular = scene_ubo_data.light.ambient =
        circe::Color(1.0, 0.5, 0.5);
    scene_ubo.push(object.program);
    object.program.setUniformBlockBinding("Scene", 0);
    scene_ubo["Scene"] = &scene_ubo_data;

    normal_map = circe::gl::Texture::fromFile(assets_path + "brickwall.jpg");
  }

  void prepareFrame(const circe::gl::ViewportDisplay &display) override {
    circe::gl::BaseApp::prepareFrame(display);
//    auto p = app_->getCamera()->getPosition();
//    circe::Light l;
//    l.type = circe::LightTypes::DIRECTIONAL;
//    l.direction = ponos::normalize(ponos::vec3(p.x, p.y, p.z));
//    shadow_map.setLight(l);
    shadow_map.render([&]() {
//      glDisable( GL_CULL_FACE );
      cartesian_plane.draw();
      object.draw();
//      glEnable( GL_CULL_FACE );
    });
  }

  void render(circe::CameraInterface *camera) override {
    drawModel(object, camera);
    drawModel(cartesian_plane, camera);
    shadow_map.depthMap().bind(GL_TEXTURE0);
    depth_buffer_view.update();
    ImGui::Begin("Shadow Map");
    texture_id = shadow_map.depthMap().textureObjectId();
    texture_id = depth_buffer_view.image.textureObjectId();
    normal_map.bind(GL_TEXTURE0);
    texture_id = normal_map.textureObjectId();
    ImGui::Image((void *) (intptr_t) (texture_id), {256, 256},
                 {0, 1}, {1, 0});
    ImGui::End();
  }

  void drawModel(Model &model, circe::CameraInterface *camera) const {
    model.program.use();
    if (model.program.hasUniform("material.kAmbient")) {
      model.program.setUniform("material.kAmbient", model.material.ambient.rgb());
      model.program.setUniform("material.kDiffuse", model.material.diffuse.rgb());
      model.program.setUniform("material.kSpecular", model.material.specular.rgb());
      model.program.setUniform("material.shininess", model.material.shininess);
    }
    model.program.setUniform("model", ponos::transpose(model.transform.matrix()));
    model.program.setUniform("view",
                             ponos::transpose(camera->getViewTransform().matrix()));
    model.program.setUniform("projection",
                             ponos::transpose(camera->getProjectionTransform().matrix()));
    model.program.setUniform("cameraPosition", camera->getPosition());
    model.program.setUniform("lightSpaceMatrix", ponos::transpose(shadow_map.light_transform().matrix()));
    model.program.setUniform("shadowMap", 0);
    shadow_map.depthMap().bind(GL_TEXTURE0);
    glEnable(GL_DEPTH_TEST);
    model.draw();
  }

  void loadObject(const ponos::Path &path) {
    ponos::RawMesh rm;
    circe::loadOBJ(path.fullName(), &rm);
    auto sphere = ponos::RawMeshes::icosphere(ponos::point3(), 1.f, 0, true, false);
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

    {
      ponos::AoS aos;
      aos.pushField<ponos::point3>("position");
      aos.pushField<ponos::vec3>("normal");
      aos.pushField<ponos::point2>("uv");
      aos.resize(3);
      aos.valueAt<ponos::point3>(0, 0) = {0, 0, 0};
      aos.valueAt<ponos::vec3>(0, 1) = {0, 0, 1};
      aos.valueAt<ponos::point2>(0, 2) = {0, 0};

      aos.valueAt<ponos::point3>(1, 0) = {10, 0, 0};
      aos.valueAt<ponos::vec3>(1, 1) = {0, 0, 1};
      aos.valueAt<ponos::point2>(1, 2) = {1, 0};

      aos.valueAt<ponos::point3>(2, 0) = {0, 10, 0};
      aos.valueAt<ponos::vec3>(2, 1) = {0, 0, 1};
      aos.valueAt<ponos::point2>(2, 2) = {0, 1};

      model.vb = aos;
      std::vector<u32> index_data = {0,1,2};
      model.ib = index_data;
      model.vao.bind();
      model.vb.bindAttributeFormats();
      return;
    }


    // prepare mesh data
    std::vector<float> vertex_data;
    std::vector<u32> index_data;
    circe::gl::setup_buffer_data_from_mesh(mesh, vertex_data, index_data);
    // describe vertex buffer
    model.vb.attributes.push<ponos::point3>("position");
    if (mesh.normalDescriptor.count)
      model.vb.attributes.push<ponos::vec3>("normal");
    if (mesh.texcoordDescriptor.count)
      model.vb.attributes.push<ponos::point2>("uv");
    // upload data
    model.vb = vertex_data;
    model.ib = index_data;
    model.ib.element_count = index_data.size() / 3;
    // bind attributes
    model.vao.bind();
    model.vb.bindAttributeFormats();
  }

  circe::gl::Texture normal_map;
  DepthBufferView depth_buffer_view;
  circe::gl::ShadowMap shadow_map;
  circe::gl::UniformBuffer scene_ubo;
  Model object, cartesian_plane;
  circe::Model mesh;
  GLuint texture_id{0};
};

int main() {
  return HelloCirce().run();
}
