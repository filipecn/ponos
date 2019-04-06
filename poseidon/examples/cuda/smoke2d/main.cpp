#include "render.h"
#include <circe/circe.h>
#include <poseidon/poseidon.h>

#define WIDTH 64
#define HEIGHT 64

class ParticleSystemModel : public circe::SceneObject {
public:
  ParticleSystemModel() {
    const char *vs = "#version 440 core\n"
                     "layout(location = 0) in vec2 position;"
                     "layout(location = 1) uniform mat4 model_view_matrix;"
                     "layout(location = 2) uniform mat4 projection_matrix;"
                     "void main(){"
                     "gl_Position = projection_matrix * model_view_matrix *"
                     " vec4(position.x, position.y, 0, 1);"
                     "}";
    const char *fs = "#version 440 core\n"
                     "out vec4 outColor;"
                     "void main() {"
                     "   outColor = vec4(1,0,0,1);"
                     "}";
    shader.reset(new circe::ShaderProgram(vs, nullptr, fs));
    shader->addVertexAttribute("position", 0);
    shader->addUniform("model_view_matrix", 1);
    shader->addUniform("projection_matrix", 2);
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    circe::BufferDescriptor bd(2, 100, GL_POINTS, GL_ARRAY_BUFFER,
                               GL_DYNAMIC_DRAW, GL_FLOAT);
    bd.addAttribute("position", 2, 0, GL_FLOAT);
    buffer.reset(new circe::VertexBuffer(nullptr, bd));
    buffer->locateAttributes(*shader.get());
  }
  ~ParticleSystemModel() {}
  void update() {
    std::vector<ponos::point2f> particles;
    for (int i = 0; i < 100; i++)
      particles.emplace_back(i / 100.0, i / 100.0);
    using namespace circe;
    buffer->resize(particles.size());
    buffer->set((const float *)particles.data());
    CHECK_GL_ERRORS;
  }
  void draw(const circe::CameraInterface *camera, ponos::Transform t) override {
    glBindVertexArray(VAO);
    shader->begin();
    shader->setUniform("model_view_matrix",
                       ponos::transpose(camera->getViewTransform().matrix()));
    shader->setUniform(
        "projection_matrix",
        ponos::transpose(camera->getProjectionTransform().matrix()));
    using namespace circe;
    CHECK_GL_ERRORS;
    glDrawArrays(GL_POINTS, 0, 100);
    CHECK_GL_ERRORS;
    shader->end();
  }

  GLuint VAO;
  std::shared_ptr<circe::ShaderProgram> shader;
  std::shared_ptr<circe::VertexBuffer> buffer;
};

int main() {
  // SIM
  poseidon::cuda::GridSmokeSolver2 solver;
  solver.setResolution(ponos::uivec2(64, 64));
  solver.setDx(0.01);
  solver.init();
  poseidon::cuda::GridSmokeInjector2::injectCircle(ponos::point2f(0.2f), .1f,
                                                   solver.densityData());
  solver.densityData().texture().updateTextureMemory();
  hermes::cuda::fill(solver.velocityData().u().texture(), 0.0f);
  hermes::cuda::fill(solver.velocityData().v().texture(), -1.0f);
  std::cerr << solver.velocityData().u().texture() << std::endl;
  std::cerr << solver.velocityData().v().texture() << std::endl;
  std::cerr << solver.densityData().texture() << std::endl;
  for (int i = 0; i < 1; i++) {
    solver.step(0.001);
    solver.densityData().texture().updateTextureMemory();
  }
  std::cerr << solver.densityData().texture() << std::endl;
  std::cerr << solver.densityData().toWorldTransform().getMatrix() << std::endl;
  std::cerr << solver.densityData().toWorldTransform()(
      hermes::cuda::point2f(0.5f));
  std::cerr << solver.densityData().toFieldTransform()(
      hermes::cuda::point2f(0.5f));
  std::cerr << solver.velocityData().u().toFieldTransform()(
                   hermes::cuda::point2f(0.5f))
            << std::endl;
  std::cerr << solver.velocityData().v().toFieldTransform()(
                   hermes::cuda::point2f(0.5f))
            << std::endl;
  // VIS
  circe::SceneApp<> app(WIDTH, HEIGHT, "", false);
  app.addViewport2D(0, 0, WIDTH, HEIGHT);
  CudaOpenGLInterop cgl(WIDTH, HEIGHT);
  circe::ScreenQuad screen;
  screen.shader->begin();
  screen.shader->setUniform("tex", 0);
  ParticleSystemModel ps;
  ps.update();
  app.scene.add(&ps);
  // app.renderCallback = [&]() {
  //   solver.step(0.001);
  //   solver.densityData().texture().updateTextureMemory();
  //   renderDensity(WIDTH, HEIGHT, solver.densityData().texture(),
  //                 cgl.bufferPointer());
  //   cgl.sendToTexture();
  //   cgl.bindTexture(GL_TEXTURE0);
  //   screen.render();
  // };
  app.run();
  return 0;
}