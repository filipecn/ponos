// Created by filipecn on 3/3/18.
#include <aergia/aergia.h>
#include <ponos/ponos.h>

#define WIDTH 800
#define HEIGHT 800

const char *fs = "#version 440 core\n"
                 "out vec4 outColor;"
                 "in vec2 texCoord;"
                 "uniform sampler2D tex;"
                 "void main() {"
                 "ivec2 texel = ivec2(gl_FragCoord.xy);"
                 "outColor = (texelFetchOffset(tex, texel, 0, ivec2(0, 0)) +"
                 "texelFetchOffset(tex, texel, 0, ivec2(1, 1)) +"
                 "texelFetchOffset(tex, texel, 0, ivec2(0, 1)) +"
                 "texelFetchOffset(tex, texel, 0, ivec2(-1, 1)) +"
                 "texelFetchOffset(tex, texel, 0, ivec2(-1, 0)) +"
                 "texelFetchOffset(tex, texel, 0, ivec2(1, 0)) +"
                 "texelFetchOffset(tex, texel, 0, ivec2(-1, -1)) +"
                 "texelFetchOffset(tex, texel, 0, ivec2(1, -1)) +"
                 "texelFetchOffset(tex, texel, 0, ivec2(0, -1))) / 9.0;"
                 "}";

int main() {
  aergia::SceneApp<> app(WIDTH, HEIGHT, "Post Effects Example");
  app.init();
  app.viewports[0].renderer->addEffect(new aergia::FXAA());
  app.viewports[0].renderer->addEffect(new aergia::PostEffect(
      new aergia::Shader(aergia::ShaderManager::instance().loadFromTexts(
          AERGIA_NO_VAO_VS, nullptr, fs))));
  std::shared_ptr<aergia::CartesianGrid> grid(
      app.scene.add<aergia::CartesianGrid>(new aergia::CartesianGrid(5)));
  app.run();
  return 0;
}