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

#include <aergia/ui/text.h>

namespace aergia {

Text::Text(const char *font) {
#ifdef FREETYPE_INCLUDED
  if (FT_Init_FreeType(&ft))
    std::cout << "ERROR::FREETYPE: Could not init FreeType Library"
              << std::endl;
  if (FT_New_Face(ft, font, 0, &ftFace)) {
    std::cout << "ERROR::FREETYPE: Failed to load font "
              << FT_New_Face(ft, font, 0, &ftFace) << std::endl;
    ;
  }
  FT_Set_Pixel_Sizes(ftFace, 0, 48);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // Disable byte-alignment restriction
  for (GLubyte c = 0; c < 128; c++) {
    // Load character glyph
    if (FT_Load_Char(ftFace, c, FT_LOAD_RENDER)) {
      std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
      continue;
    }
    aergia::TextureAttributes attributes;
    attributes.internalFormat = GL_RED;
    attributes.target = GL_TEXTURE_2D;
    attributes.width = ftFace->glyph->bitmap.width;
    attributes.height = ftFace->glyph->bitmap.rows;
    attributes.format = GL_RED;
    attributes.type = GL_UNSIGNED_BYTE;
    attributes.data = ftFace->glyph->bitmap.buffer;
    aergia::TextureParameters parameters;
    // Now store character for later use
    Character character = {
        new aergia::Texture(attributes, parameters),
        ponos::ivec2(ftFace->glyph->bitmap.width, ftFace->glyph->bitmap.rows),
        ponos::ivec2(ftFace->glyph->bitmap_left, ftFace->glyph->bitmap_top),
        static_cast<GLuint>(ftFace->glyph->advance.x)};
    characters.insert(std::pair<GLchar, Character>(c, character));
  }
  FT_Done_Face(ftFace);
  FT_Done_FreeType(ft);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  quad.shader.reset(
      new aergia::Shader(aergia::ShaderManager::instance().loadFromTexts(
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
          "}")));
  quad.shader->addVertexAttribute("position");
  quad.shader->addVertexAttribute("texcoord");
#else
  UNUSED_VARIABLE(font);
#endif
}
void Text::render(std::string s, GLfloat x, GLfloat y, GLfloat scale,
                  aergia::Color c) {
#ifdef FREETYPE_INCLUDED
  quad.shader->setUniform("textColor", ponos::vec4(c.r, c.g, c.b, c.a));
  quad.shader->setUniform(
      "projection", ponos::transpose(ponos::ortho(0, 800, 0, 800).matrix()));
  quad.shader->setUniform("tex", 0);
  std::string::const_iterator it;
  for (it = s.begin(); it != s.end(); it++) {
    const Character ch = characters.at(*it);
    GLfloat xpos = x + ch.bearing[0] * scale;
    GLfloat ypos = y - (ch.size[1] - ch.bearing[1]) * scale;
    GLfloat w = ch.size[0] * scale;
    GLfloat h = ch.size[1] * scale;
    ch.texture->bind(GL_TEXTURE0);
    quad.set(ponos::Point2(xpos, ypos), ponos::Point2(xpos + w, ypos + h));
    quad.draw();
    // Now advance cursors for next glyph (note that advance is number of 1/64
    // pixels)
    x += (ch.advance >> 6) *
         scale; // Bitshift by 6 to get value in pixels (2^6 = 64)
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  quad.shader->end();
#else
  UNUSED_VARIABLE(s);
  UNUSED_VARIABLE(x);
  UNUSED_VARIABLE(y);
  UNUSED_VARIABLE(scale);
  UNUSED_VARIABLE(c);
  std::cerr << "FREETYPE not included!\n";
#endif
}

void Text::render(std::string s, const ponos::Point3 &p, GLfloat scale,
                  aergia::Color c) {
  ponos::Point3 sp = GraphicsDisplay::instance().normDevCoordToViewCoord(p);
  render(s, sp.x, sp.y, scale, c);
}

} // aergia namespace
