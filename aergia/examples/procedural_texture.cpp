#include <aergia/aergia.h>

aergia::SceneApp<> app(800, 800, "");
aergia::ProceduralTexture *pt;
const char *vs = "#version 440 core\n"
                 "in vec2 position;"
                 "in vec2 texcoord;"
                 "uniform mat4 proj;"
                 "out vec2 Texcoord;"
                 "void main() {"
                 "   Texcoord = texcoord;"
                 "   gl_Position = proj * vec4(position, 0.0, 1.0);"
                 "}";

const char *fs = "#version 440 core\n"
                 "in vec2 Texcoord;"
                 "out vec4 outColor;"
                 "uniform sampler2D tex;"
                 "void main() {"
                 "   outColor = texture(tex, Texcoord);"
                 "}";

const char *vs2 = AERGIA_NO_VAO_VS;/*"#version 440 core\n"
                  "out vec2 texCoord;"
                  "void main()"
                  "{"
                  "    float x = -1.0 + float((gl_VertexID & 1) << 2);"
                  "    float y = -1.0 + float((gl_VertexID & 2) << 1);"
                  "    texCoord.x = (x+1.0)*0.5;"
                  "    texCoord.y = (y+1.0)*0.5;"
                  "    gl_Position = vec4(x, y, 0, 1);"
                  "}";*/

const char *fs2 = "#version 440 core\n"
                  "out vec4 outColor;"
                  "void main() {"
                  "outColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);"
                  "}";

class QQuad : public aergia::SceneMesh {
public:
  QQuad() {
    rawMesh = new ponos::RawMesh();
    rawMesh->meshDescriptor.elementSize = 4;
    rawMesh->meshDescriptor.count = 1;
    rawMesh->vertexDescriptor.elementSize = 2;
    rawMesh->vertexDescriptor.count = 4;
    rawMesh->texcoordDescriptor.elementSize = 2;
    rawMesh->texcoordDescriptor.count = 4;
    rawMesh->vertices = std::vector<float>({-1, -1, 1, -1, 1, 1, -1, 1});
    rawMesh->texcoords = std::vector<float>({0, 0, 1, 0, 1, 1, 0, 1});
    rawMesh->indices.resize(4);
    for (int i = 0; i < 4; i++)
      rawMesh->indices[i].vertexIndex = rawMesh->indices[i].texcoordIndex = i;
    rawMesh->splitIndexData();
    rawMesh->buildInterleavedData();
    glGenVertexArrays(1, &VAO);
    setupVertexBuffer(/*GL_TRIANGLES, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW*/);
    setupIndexBuffer();
  }
  ~QQuad() {}
  void set(const ponos::Point2 &pm, const ponos::Point2 &pM) {
    rawMesh->interleavedData[0] = pm.x;
    rawMesh->interleavedData[1] = pm.y;
    rawMesh->interleavedData[4] = pM.x;
    rawMesh->interleavedData[5] = pm.y;
    rawMesh->interleavedData[8] = pM.x;
    rawMesh->interleavedData[9] = pM.y;
    rawMesh->interleavedData[12] = pm.x;
    rawMesh->interleavedData[13] = pM.y;
    glBindVertexArray(VAO);
    vb->set(&rawMesh->interleavedData[0]);
    glBindVertexArray(0);
  }
  void draw() const override {
    glBindVertexArray(VAO);
    vb->bind();
    ib->bind();
    shader->begin(vb.get());
    glDrawElements(GL_QUADS, ib->bufferDescriptor.elementCount, GL_UNSIGNED_INT,
                   0);
    shader->end();
    glBindVertexArray(0);
  }

  aergia::Shader *shader;

private:
  ponos::Point2 pMin, pMax;
  GLuint VAO;
};

int main() {
  WIN32CONSOLE();
  app.init();
  glEnable(GL_TEXTURE_2D);
  aergia::TextureAttributes attributes;
  attributes.target = GL_TEXTURE_2D;
  attributes.width = 16;
  attributes.height = 16;
  attributes.type = GL_UNSIGNED_BYTE;
  attributes.internalFormat = GL_RGBA8;
  attributes.format = GL_RGBA;
  aergia::TextureParameters parameters;
  aergia::ShaderManager &sm = aergia::ShaderManager::instance();
  aergia::Shader s2(sm.loadFromTexts(vs2, nullptr, fs2));
  pt = new aergia::ProceduralTexture(attributes, parameters);
  pt->render([&s2]() {
    s2.begin();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    s2.end();
  });
  app.viewports[0].camera.reset(new aergia::Camera2D());
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())
      ->setPos(ponos::vec2());
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())->setZoom(1.5f);
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())
      ->resize(800, 800);
  QQuad quad;
  quad.shader = new aergia::Shader(sm.loadFromTexts(vs, nullptr, fs));
  quad.shader->addVertexAttribute("position");
  quad.shader->addVertexAttribute("texcoord");
  quad.shader->setUniform("proj",
                          app.viewports[0].camera->getTransform().matrix());
  quad.shader->setUniform("tex", 0);
  pt->bind(GL_TEXTURE0);
  app.scene.add(&quad);
  app.run();
  return 0;
}
