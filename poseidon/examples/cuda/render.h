#include <circe/circe.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <poseidon/poseidon.h>

class CudaOpenGLInterop {
public:
  CudaOpenGLInterop() {}
  void set(unsigned int w, unsigned int h) {
    width = w;
    height = h;
    using namespace hermes::cuda;
    unsigned int size_tex_data = sizeof(GLubyte) * width * height * 4;
    CUDA_CHECK(cudaMalloc(&cuda_dev_render_buffer, size_tex_data));
    circe::TextureAttributes ta;
    ta.width = width;
    ta.height = height;
    ta.internalFormat = GL_RGBA8;
    ta.format = GL_RGBA;
    ta.type = GL_UNSIGNED_BYTE;
    ta.target = GL_TEXTURE_2D;
    circe::TextureParameters tp;
    tp[GL_TEXTURE_MIN_FILTER] = GL_NEAREST;
    tp[GL_TEXTURE_MAG_FILTER] = GL_NEAREST;
    texture.set(ta, tp);
    // Register this texture with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &cuda_tex_resource, texture.textureObjectId(), GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard));
    using namespace circe;
    CHECK_GL_ERRORS;
  }

  ~CudaOpenGLInterop() {
    if (cuda_dev_render_buffer)
      cudaFree(cuda_dev_render_buffer);
  }

  void sendToTexture() {
    using namespace hermes::cuda;
    // We want to copy cuda_dev_render_buffer data to the texture
    // Map buffer objects to get CUDA device pointers
    cudaArray *texture_ptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texture_ptr,
                                                     cuda_tex_resource, 0, 0));

    int num_texels = width * height * depth;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    CUDA_CHECK(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer,
                                 size_tex_data, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));
  }

  void bindTexture(GLenum t) { texture.bind(t); }
  template <typename T> T *bufferPointer() {
    return (T *)cuda_dev_render_buffer;
  }

private:
  circe::Texture texture;
  unsigned int width = 0, height = 0, depth = 1;
  void *cuda_dev_render_buffer = nullptr;
  struct cudaGraphicsResource *cuda_tex_resource = nullptr;
};

void renderDistances(hermes::cuda::RegularGrid2Df &in, unsigned int *out,
                     hermes::cuda::Color a, hermes::cuda::Color b);

template <hermes::cuda::MemoryLocation L>
class Levelset2Model : public circe::SceneObject {
public:
  Levelset2Model(poseidon::cuda::LevelSet2<L> &ls) : ls_(ls) {
    circe::BufferDescriptor vertex_desc =
        circe::create_vertex_buffer_descriptor(2, 0, GL_LINES);
    vertex_desc.addAttribute(std::string("position"), 2, 0, GL_FLOAT);
    circe::BufferDescriptor index_desc = circe::create_index_buffer_descriptor(
        2, 0, ponos::GeometricPrimitiveType::LINES);
    m_.mesh().setDescription(vertex_desc, index_desc);
    update();
    auto vs = std::string(SHADERS_PATH) + "/isoline.vert";
    auto fs = std::string(SHADERS_PATH) + "/isoline.frag";
    shader = circe::createShaderProgramPtr(
        circe::ShaderManager::instance().loadFromFiles(
            {vs.c_str(), fs.c_str()}));
    shader->addVertexAttribute("position", 0);
    shader->addUniform("model", 1);
    shader->addUniform("view", 2);
    shader->addUniform("projection", 3);
    shader->addUniform("color", 4);
    m_.setShader(shader);
    m_.draw_callback = [&](circe::ShaderProgram *s,
                           const circe::CameraInterface *camera,
                           ponos::Transform t) {
      s->begin();
      s->setUniform("color", color);
      s->setUniform("model", ponos::transpose(t.matrix()));
      s->setUniform("view",
                    ponos::transpose(camera->getViewTransform().matrix()));
      s->setUniform(
          "projection",
          ponos::transpose(camera->getProjectionTransform().matrix()));
    };
    // cgl.set(ls_.grid().resolution().x, ls_.grid().resolution().y);
  }
  void update() {
    ls_.isoline(vertices_, indices_);
    if (!indices_.size())
      // TODO: clean current buffer
      return;
    // TODO: use cuda opengl interop to avoid memory transference
    rm_.primitiveType = ponos::GeometricPrimitiveType::LINES;
    rm_.meshDescriptor.count = indices_.size() / 2;
    rm_.meshDescriptor.elementSize = 2;
    rm_.positionDescriptor.count = vertices_.size() / 2;
    rm_.positionDescriptor.elementSize = 2;
    hermes::cuda::memcpy(rm_.positions, vertices_);
    hermes::cuda::memcpy(rm_.positionsIndices, indices_);
    m_.mesh().update(&rm_.positions[0], rm_.positionDescriptor.count,
                     &rm_.positionsIndices[0], rm_.meshDescriptor.count);
  }
  void draw(const circe::CameraInterface *camera, ponos::Transform t) override {
    // renderDistances(ls_.grid(), cgl.bufferPointer<unsigned int>(),
    //                 hermes::cuda::Color(1, 0, 0, 1),
    //                 hermes::cuda::Color(1, 0, 1, 1));
    // glEnable(GL_DEPTH_TEST);
    // cgl.sendToTexture();
    // cgl.bindTexture(GL_TEXTURE0);
    // dist_.draw(camera, t);
    // glDisable(GL_DEPTH_TEST);
    m_.draw(camera, t);
  }

  circe::Color color = circe::Color::White();

private:
  // data
  poseidon::cuda::LevelSet2<L> &ls_;
  hermes::cuda::MemoryBlock1Df vertices_;
  hermes::cuda::MemoryBlock1Du indices_;
  // vis
  CudaOpenGLInterop cgl;
  ponos::RawMesh rm_;
  circe::SceneDynamicMeshObject m_;
  circe::ShaderProgramPtr shader;
  circe::Quad dist_;
};

template <hermes::cuda::MemoryLocation L>
class Levelset3Model : public circe::SceneObject {
public:
  Levelset3Model(poseidon::cuda::LevelSet3<L> &ls) : ls_(ls) {
    circe::BufferDescriptor vertex_desc =
        circe::create_vertex_buffer_descriptor(6, 0, GL_TRIANGLES);
    vertex_desc.addAttribute(std::string("position"), 3, 0, GL_FLOAT);
    vertex_desc.addAttribute(std::string("normal"), 3, 3 * sizeof(float),
                             GL_FLOAT);
    circe::BufferDescriptor index_desc = circe::create_index_buffer_descriptor(
        3, 0, ponos::GeometricPrimitiveType::TRIANGLES);
    m_.mesh().setDescription(vertex_desc, index_desc);
    update();
    auto vs = std::string(SHADERS_PATH) + "/isosurface.vert";
    auto fs = std::string(SHADERS_PATH) + "/isosurface.frag";
    shader = circe::createShaderProgramPtr(
        circe::ShaderManager::instance().loadFromFiles(
            {vs.c_str(), fs.c_str()}));
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
    m_.setShader(shader);
    m_.draw_callback = [](circe::ShaderProgram *s,
                          const circe::CameraInterface *camera,
                          ponos::Transform t) {
      s->begin();
      s->setUniform("Light.position", ponos::vec3(0, -1, 1));
      s->setUniform("Light.ambient", ponos::vec3(1, 1, 1));
      s->setUniform("Light.diffuse", ponos::vec3(1, 1, 1));
      s->setUniform("Light.specular", ponos::vec3(1, 1, 1));
      s->setUniform("Material.kAmbient", ponos::vec3(0.5, 0.5, 0.5));
      s->setUniform("Material.kDiffuse", ponos::vec3(1.0, 0.5, 0.5));
      s->setUniform("Material.kSpecular", ponos::vec3(0.8, 1, 1));
      s->setUniform("Material.shininess", 100.f);
      s->setUniform("model", ponos::transpose(t.matrix()));
      s->setUniform("view",
                    ponos::transpose(camera->getViewTransform().matrix()));
      s->setUniform(
          "projection",
          ponos::transpose(camera->getProjectionTransform().matrix()));
      s->setUniform("cameraPosition", camera->getPosition());
    };
  }
  void update() {
    std::cerr << "update level set:\n";
    ls_.isosurface(vertices_, indices_, 0.f, &normals_);
    std::cerr << indices_.size() << std::endl;
    if (!indices_.size())
      // TODO: clean current buffer
      return;

    // TODO: use cuda opengl interop to avoid memory transference
    rm_.primitiveType = ponos::GeometricPrimitiveType::TRIANGLES;
    rm_.meshDescriptor.count = indices_.size() / 3;
    rm_.meshDescriptor.elementSize = 3;
    rm_.normalDescriptor.count = vertices_.size() / 3;
    rm_.normalDescriptor.elementSize = 3;
    rm_.positionDescriptor.count = vertices_.size() / 3;
    rm_.positionDescriptor.elementSize = 3;
    hermes::cuda::memcpy(rm_.positions, vertices_);
    hermes::cuda::memcpy(rm_.normals, normals_);
    hermes::cuda::memcpy(rm_.positionsIndices, indices_);
    rm_.buildInterleavedData();
    // std::cerr << rm_ << std::endl;
    m_.mesh().update(&rm_.interleavedData[0], rm_.positionDescriptor.count,
                     &rm_.positionsIndices[0], rm_.meshDescriptor.count);
  }
  void draw(const circe::CameraInterface *camera, ponos::Transform t) override {
    glEnable(GL_DEPTH_TEST);
    m_.draw(camera, t);
  }

private:
  // data
  poseidon::cuda::LevelSet3<L> &ls_;
  hermes::cuda::MemoryBlock1Df vertices_, normals_;
  hermes::cuda::MemoryBlock1Du indices_;
  // vis
  ponos::RawMesh rm_;
  circe::SceneDynamicMeshObject m_;
  circe::ShaderProgramPtr shader;
};

using LevelSet2ModelD = Levelset2Model<hermes::cuda::MemoryLocation::DEVICE>;
using LevelSet2ModelH = Levelset2Model<hermes::cuda::MemoryLocation::HOST>;
using LevelSet3ModelD = Levelset3Model<hermes::cuda::MemoryLocation::DEVICE>;
using LevelSet3ModelH = Levelset3Model<hermes::cuda::MemoryLocation::HOST>;

template <hermes::cuda::MemoryLocation L>
class LevelSet3DistancesModel : public circe::SceneObject {
public:
  LevelSet3DistancesModel(poseidon::cuda::LevelSet3<L> &ls) : ls_(ls) {
    glGenVertexArrays(1, &VAO_);
    buffer_.bufferDescriptor =
        circe::create_vertex_buffer_descriptor(6, 0, GL_POINTS);
    buffer_.bufferDescriptor.addAttribute(std::string("position"), 3, 0,
                                          GL_FLOAT);
    buffer_.bufferDescriptor.addAttribute(std::string("color"), 3,
                                          3 * sizeof(float), GL_FLOAT);
    auto vs = std::string(SHADERS_PATH) + "/point3.vert";
    auto fs = std::string(SHADERS_PATH) + "/point3.frag";
    shader_ = circe::createShaderProgramPtr(
        circe::ShaderManager::instance().loadFromFiles(
            {vs.c_str(), fs.c_str()}));
    shader_->addVertexAttribute("position", 0);
    shader_->addVertexAttribute("color", 1);
    shader_->addUniform("model", 2);
    shader_->addUniform("view", 3);
    shader_->addUniform("projection", 4);
    font_id_ = circe::FontManager::loadFromFile(
        (std::string(ASSETS_PATH) + "/arial.ttf").c_str());
  }
  void update() {
    labels_.clear();
    size_t count = ls_.grid().resolution().x * ls_.grid().resolution().y *
                   ls_.grid().resolution().z;
    if (vertex_and_color_.size() != count * 6)
      vertex_and_color_.resize(count * 6);
    size_t i = 0;
    hermes::cuda::RegularGrid3Hf h_grid(ls_.grid());
    for (auto e : h_grid.accessor()) {
      auto p = e.worldPosition();
      vertex_and_color_[i * 6 + 0] = p.x;
      vertex_and_color_[i * 6 + 1] = p.y;
      vertex_and_color_[i * 6 + 2] = p.z;
      vertex_and_color_[i * 6 + 3] = fabs(e.value);
      vertex_and_color_[i * 6 + 4] = fabs(e.value);
      vertex_and_color_[i * 6 + 5] = fabs(e.value);
      labels_.emplace_back(font_id_);
      labels_[i].setText(ponos::concat(e.value));
      labels_[i].text_size = 0.00004;
      labels_[i].text_color = circe::COLOR_RED;
      labels_[i].position = ponos::point3(p.x, p.y, p.z);
      i++;
    }
    buffer_.bufferDescriptor.elementCount = count;
    buffer_.bufferDescriptor.elementSize = 6;
    glBindVertexArray(VAO_);
    buffer_.set(vertex_and_color_.data());
    glBindVertexArray(0);
  }
  void draw(const circe::CameraInterface *camera, ponos::Transform t) override {
    if (!this->visible)
      return;
    glBindVertexArray(VAO_);
    buffer_.bind();
    shader_->begin();
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    buffer_.locateAttributes(*shader_.get());
    shader_->setUniform("model", ponos::transpose(t.matrix()));
    shader_->setUniform("view",
                        ponos::transpose(camera->getViewTransform().matrix()));
    shader_->setUniform(
        "projection",
        ponos::transpose(camera->getProjectionTransform().matrix()));
    glDrawArrays(buffer_.bufferDescriptor.elementType, 0,
                 buffer_.bufferDescriptor.elementCount);
    shader_->end();
    glBindVertexArray(0);
    int count = 0;
    for (auto &l : labels_) {
      if (true ||
          // camera->getFrustum().isInside(l.position)) {
          ponos::dot(camera->getTarget() - camera->getPosition(),
                     l.position - camera->getPosition()) >= 0.3) {
        // &&
        // ponos::distance(l.position, camera->getPosition()) < 10.f / 32.f) {
        l.draw(camera, t);
        count++;
      }
    }
    std::cerr << count << std::endl;
  }

private:
  // data
  poseidon::cuda::LevelSet3<L> &ls_;
  // vis
  GLuint VAO_;
  circe::VertexBuffer buffer_;
  std::vector<float> vertex_and_color_;
  circe::ShaderProgramPtr shader_;
  // text
  int font_id_ = -1;
  std::vector<circe::TextObject> labels_;
};

using LevelSet3DistancesModelD =
    LevelSet3DistancesModel<hermes::cuda::MemoryLocation::DEVICE>;
using LevelSet3DistancesModelH =
    LevelSet3DistancesModel<hermes::cuda::MemoryLocation::HOST>;

template <hermes::cuda::MemoryLocation L>
class ParticleSystem2Model : public circe::SceneObject {
public:
  ParticleSystem2Model(poseidon::cuda::ParticleSystem2<L> &ps)
      : particles_(ps) {
    circe::BufferDescriptor vertex_desc =
        circe::create_vertex_buffer_descriptor(2, 0, GL_POINTS);
    vertex_desc.addAttribute(std::string("position"), 2, 0, GL_FLOAT);
    circe::BufferDescriptor index_desc = circe::create_index_buffer_descriptor(
        2, 0, ponos::GeometricPrimitiveType::POINTS);
    m_.mesh().setDescription(vertex_desc, index_desc);
    update();
    auto vs = std::string(SHADERS_PATH) + "/particle2.vert";
    auto fs = std::string(SHADERS_PATH) + "/particle2.frag";
    shader = circe::createShaderProgramPtr(
        circe::ShaderManager::instance().loadFromFiles(
            {vs.c_str(), fs.c_str()}));
    shader->addVertexAttribute("position", 0);
    shader->addUniform("model", 1);
    shader->addUniform("view", 2);
    shader->addUniform("projection", 3);
    shader->addUniform("color", 4);
    m_.setShader(shader);
    m_.draw_callback = [](circe::ShaderProgram *s,
                          const circe::CameraInterface *camera,
                          ponos::Transform t) {
      s->begin();
      s->setUniform("color", ponos::vec4(1, 1, 1, 1));
      s->setUniform("model", ponos::transpose(t.matrix()));
      s->setUniform("view",
                    ponos::transpose(camera->getViewTransform().matrix()));
      s->setUniform(
          "projection",
          ponos::transpose(camera->getProjectionTransform().matrix()));
    };
  }
  void update() {
    if (!indices_.size())
      // TODO: clean current buffer
      return;
    // TODO: use cuda opengl interop to avoid memory transference
    rm_.primitiveType = ponos::GeometricPrimitiveType::POINTS;
    rm_.meshDescriptor.count = indices_.size() / 1;
    rm_.meshDescriptor.elementSize = 1;
    rm_.positionDescriptor.count = vertices_.size() / 2;
    rm_.positionDescriptor.elementSize = 2;
    hermes::cuda::memcpy(rm_.positions, vertices_);
    hermes::cuda::memcpy(rm_.positionsIndices, indices_);
    m_.mesh().update(&rm_.positions[0], rm_.positionDescriptor.count,
                     &rm_.positionsIndices[0], rm_.meshDescriptor.count);
  }
  void draw(const circe::CameraInterface *camera, ponos::Transform t) override {
  }

private:
  // data
  poseidon::cuda::ParticleSystem2<L> &particles_;
  hermes::cuda::MemoryBlock1Df vertices_;
  hermes::cuda::MemoryBlock1Du indices_;
  // vis
  ponos::RawMesh rm_;
  circe::SceneDynamicMeshObject m_;
  circe::ShaderProgramPtr shader;
};