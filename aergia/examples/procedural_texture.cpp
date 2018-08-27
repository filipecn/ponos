#include <aergia/aergia.h>

aergia::SceneApp<> app(800, 800, "", false);
aergia::RenderTexture *pt;
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
  pt = new aergia::RenderTexture(attributes, parameters);
  pt->render([&s2]() {
    s2.begin();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    aergia::CHECK_GL_ERRORS;
    s2.end();
  });
  app.addViewport2D(0,0,800,800);
  static_cast<aergia::UserCamera2D *>(app.viewports[0].camera.get())
      ->setPosition(ponos::Point3());
  static_cast<aergia::UserCamera2D *>(app.viewports[0].camera.get())->setZoom(1.5f);
  static_cast<aergia::UserCamera2D *>(app.viewports[0].camera.get())
      ->resize(800, 800);
  aergia::Quad quad;
  quad.shader.reset(new aergia::Shader(sm.loadFromTexts(vs, nullptr, fs)));
  quad.shader->addVertexAttribute("position");
  quad.shader->addVertexAttribute("texcoord");
//  quad.shader->setUniform("proj", aergia::glGetMVPTransform().matrix());
//                          app.viewports[0].camera->getTransform().matrix());
  quad.shader->setUniform("tex", 0);
  pt->bind(GL_TEXTURE0);
  app.scene.add(&quad);
  app.run();
  return 0;
}
