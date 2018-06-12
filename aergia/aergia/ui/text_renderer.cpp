/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
*/

#include <aergia/ui/text_renderer.h>
#include <aergia/ui/font_manager.h>

namespace aergia {

TextRenderer::TextRenderer(float scale, Color c, size_t id)
    : fontId(id), scale(scale), textColor(c) {
  dynamicScale_ = 1.f;
  dynamicColor_ = COLOR_BLACK;
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  static int sid = aergia::ShaderManager::instance().loadFromTexts(
      "#version 440 core\n"
          "in vec2 position;"
          "in vec2 texcoord;"
          "out vec2 TexCoords;"
          "uniform mat4 projection;"
          "void main() {"
          "  TexCoords = texcoord;"
          "  gl_Position = projection * vec4(position, 0.0, 1.0);"
          "}",
      nullptr,
      "#version 440 core\n"
          "in vec2 TexCoords;"
          "out vec4 color;"
          "uniform sampler2D text;"
          "uniform vec4 textColor;"
          "void main() {"
          "  vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);"
          "  color = textColor * sampled;"
          "}");
  quad_.shader.reset(new aergia::Shader(sid));
  quad_.shader->addVertexAttribute("position");
  quad_.shader->addVertexAttribute("texcoord");
}

void TextRenderer::render(std::string s, GLfloat x, GLfloat y, GLfloat scale,
                          aergia::Color c) {
  UNUSED_VARIABLE(x);
  UNUSED_VARIABLE(y);
  UNUSED_VARIABLE(scale);
  atlas.setText(s);
  atlas.mesh->bind();
  atlas.mesh->vertexBuffer()->locateAttributes(*quad_.shader.get());
  atlas.texture.bind(GL_TEXTURE0);
  quad_.shader->begin();
  quad_.shader->setUniform("textColor", ponos::vec4(c.r, c.g, c.b, c.a));
  quad_.shader->setUniform(
      "projection", ponos::transpose(ponos::ortho(0, 800, 0, 800).matrix()));
  quad_.shader->setUniform("tex", 0);
  aergia::CHECK_GL_ERRORS;
  auto ib = atlas.mesh->indexBuffer();
  glDrawElements(ib->bufferDescriptor.elementType,
                 ib->bufferDescriptor.elementCount *
                     ib->bufferDescriptor.elementSize,
                 ib->bufferDescriptor.dataType, 0);
  aergia::CHECK_GL_ERRORS;
/*
  std::string::const_iterator it;
  auto font = FontManager::instance().fontTexture(fontId);
  for (it = s.begin(); it != s.end(); it++) {
    auto ch = font[*it];
    GLfloat xpos = x + ch.bearing[0] * scale;
    GLfloat ypos = y - (ch.size[1] - ch.bearing[1]) * scale;
    GLfloat w = ch.size[0] * scale;
    GLfloat h = ch.size[1] * scale;
    ch.texture->bind(GL_TEXTURE0);
    quad_.set(ponos::Point2(xpos, ypos), ponos::Point2(xpos + w, ypos + h));
    quad_.draw();
    // Now advance cursors for next glyph (note that advance is number of 1/64
    // pixels)
    // Bitshift by 6 to get value in pixels (2^6 = 64)
    x += (ch.advance >> 6) * scale;
  }
  */
//  glBindTexture(GL_TEXTURE_2D, 0);
  quad_.shader->end();
}

void TextRenderer::render(std::string s, const ponos::Point3 &p, GLfloat scale,
                          aergia::Color c) {
  auto &gd = GraphicsDisplay::instance();
  ponos::Point3 sp = glGetMVPTransform()(p);
  sp = gd.normDevCoordToViewCoord(sp);
  render(s, sp.x, sp.y, scale, c);
}

TextRenderer &TextRenderer::at(const ponos::Point3 &p) {
  auto &gd = GraphicsDisplay::instance();
  ponos::Point3 sp = glGetMVPTransform()(p);
  sp = gd.normDevCoordToViewCoord(sp);
  dynamicPosition_.x = sp.x;
  dynamicPosition_.y = sp.y;
  return *this;
}

TextRenderer &TextRenderer::at(const ponos::Point2 &p) {
  dynamicPosition_ = p;
  return *this;
}

TextRenderer &TextRenderer::withScale(float s) {
  dynamicScale_ = s;
  return *this;
}

TextRenderer &TextRenderer::withColor(Color c) {
  dynamicColor_ = c;
  return *this;
}

TextRenderer &TextRenderer::operator<<(TextRenderer &tr) {
  return tr;
}

TextRenderer::TextRenderer(const char *filename) : TextRenderer() {
  atlas.loadFont(filename);
}

} // aergia namespace
