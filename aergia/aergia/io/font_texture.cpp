// Created by filipecn on 3/29/18.
/*
 * Copyright (c) 2018 FilipeCN
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

#include "font_texture.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

namespace aergia {

FontTexture::FontTexture() = default;

FontTexture::~FontTexture() = default;

void FontTexture::addCharacter(GLubyte c,
                               ponos::ivec2 s,
                               aergia::TextureAttributes a,
                               aergia::TextureParameters p,
                               ponos::ivec2 bearing,
                               GLuint advance) {
  Character character;
  character.texture.reset(new Texture(a, p));
  character.size = s;
  character.bearing = bearing;
  character.advance = advance;
  characters.insert(std::pair<GLchar, Character>(c, character));
}

const FontTexture::Character &FontTexture::operator[](GLubyte c) const {
  auto it = characters.find(c);
  return it->second;
}

void FontAtlas::loadFont(const char *path) {
  auto fontData = ponos::readFile(path);
  std::unique_ptr<uchar[]> atlasData(new uchar[font.atlasWidth * font.atlasHeight]);

  font.charInfo.reset(new stbtt_packedchar[font.charCount]);

  stbtt_pack_context context;
  if (!stbtt_PackBegin(&context, atlasData.get(), font.atlasWidth, font.atlasHeight, 0, 1, nullptr)) {
    std::cerr << "Failed to initialize font " << path;
    return;
  }

  stbtt_PackSetOversampling(&context, font.oversampleX, font.oversampleY);
  if (!stbtt_PackFontRange(&context,
                           fontData.data(),
                           0,
                           font.size,
                           font.firstChar,
                           font.charCount,
                           font.charInfo.get())) {
    std::cerr << "Failed to pack font\n";
  }

  stbtt_PackEnd(&context);

  glGenTextures(1, &font.texture);
  glBindTexture(GL_TEXTURE_2D, font.texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D,
               0,
               GL_RGB,
               font.atlasWidth,
               font.atlasHeight,
               0,
               GL_RED,
               GL_UNSIGNED_BYTE,
               atlasData.get());
  glHint(GL_GENERATE_MIPMAP_HINT, GL_NICEST);
  glGenerateMipmap(GL_TEXTURE_2D);
}

FontAtlas::Glyph FontAtlas::getGlyph(uint character, float offsetX, float offsetY) {
  stbtt_aligned_quad quad;

  stbtt_GetPackedQuad(font.charInfo.get(), font.atlasWidth, font.atlasHeight,
                      character - font.firstChar, &offsetX, &offsetY, &quad, 1);
  auto xmin = quad.x0;
  auto xmax = quad.x1;
  auto ymin = -quad.y1;
  auto ymax = -quad.y0;

  Glyph info{};
  info.offsetX = offsetX;
  info.offsetY = offsetY;
  info.positions[0] = ponos::vec3(xmin, ymin, 0);
  info.positions[1] = ponos::vec3(xmin, ymax, 0);
  info.positions[2] = ponos::vec3(xmax, ymax, 0);
  info.positions[3] = ponos::vec3(xmax, ymin, 0);
  info.uvs[0] = ponos::vec2(quad.s0, quad.t1);
  info.uvs[1] = ponos::vec2(quad.s0, quad.t0);
  info.uvs[2] = ponos::vec2(quad.s1, quad.t0);
  info.uvs[3] = ponos::vec2(quad.s1, quad.t1);

  return info;
}

void FontAtlas::setText(std::string text) {
  std::vector<ponos::vec3> vertices;
  std::vector<ponos::vec2> uvs;
  std::vector<uint> indexes;

  uint16_t lastIndex = 0;
  float offsetX = 0, offsetY = 0;
  for (auto c : text) {
    const auto glyphInfo = getGlyph(c, offsetX, offsetY);
    offsetX = glyphInfo.offsetX;
    offsetY = glyphInfo.offsetY;

    vertices.emplace_back(glyphInfo.positions[0]);
    vertices.emplace_back(glyphInfo.positions[1]);
    vertices.emplace_back(glyphInfo.positions[2]);
    vertices.emplace_back(glyphInfo.positions[3]);
    uvs.emplace_back(glyphInfo.uvs[0]);
    uvs.emplace_back(glyphInfo.uvs[1]);
    uvs.emplace_back(glyphInfo.uvs[2]);
    uvs.emplace_back(glyphInfo.uvs[3]);
    indexes.push_back(lastIndex);
    indexes.push_back(lastIndex + 1);
    indexes.push_back(lastIndex + 2);
    indexes.push_back(lastIndex);
    indexes.push_back(lastIndex + 2);
    indexes.push_back(lastIndex + 3);

    lastIndex += 4;
  }

  glGenVertexArrays(1, &rotatingLabel.vao);
  glBindVertexArray(rotatingLabel.vao);

  glGenBuffers(1, &rotatingLabel.vertexBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, rotatingLabel.vertexBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(ponos::vec3) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);

  glGenBuffers(1, &rotatingLabel.uvBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, rotatingLabel.uvBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(ponos::vec2) * uvs.size(), uvs.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(1);

  rotatingLabel.indexElementCount = indexes.size();
  glGenBuffers(1, &rotatingLabel.indexBuffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rotatingLabel.indexBuffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               sizeof(uint16_t) * rotatingLabel.indexElementCount,
               indexes.data(),
               GL_STATIC_DRAW);
}

void FontAtlas::render() {
  glBindVertexArray(rotatingLabel.vao);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rotatingLabel.indexBuffer);
  glDrawElements(GL_TRIANGLES, rotatingLabel.indexElementCount, GL_UNSIGNED_SHORT, nullptr);
}

}