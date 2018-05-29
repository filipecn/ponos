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

#ifndef AERGIA_FONT_TEXTURE_H
#define AERGIA_FONT_TEXTURE_H

#include <ponos/ponos.h>
#include <aergia/io/texture.h>
#include <aergia/scene/quad.h>
#include <stb_truetype.h>
#include <memory>

namespace aergia {

class FontAtlas {
 public:
  struct {
    const size_t size = 40;
    const size_t atlasWidth = 1024;
    const size_t atlasHeight = 1024;
    const size_t oversampleX = 2;
    const size_t oversampleY = 2;
    const size_t firstChar = ' ';
    const size_t charCount = '~' - ' ';
    std::unique_ptr<stbtt_packedchar[]> charInfo;
    GLuint texture = 0;
  } font;

  struct {
    GLuint vao = 0;
    GLuint vertexBuffer = 0;
    GLuint uvBuffer = 0;
    GLuint indexBuffer = 0;
    uint16_t indexElementCount = 0;
    float angle = 0;
  } rotatingLabel;

  struct Glyph {
    ponos::vec3 positions[4];
    ponos::vec2 uvs[4];
    float offsetX = 0;
    float offsetY = 0;
  };

  void loadFont(const char *path);
  Glyph getGlyph(uint character, float offsetX, float offsetY);
  void setText(std::string text);
  void render();
};

class FontTexture {
 public:
  struct Character {
    std::shared_ptr<aergia::Texture> texture;  //!< the glyph texture
    ponos::ivec2 size;        //!< size of glyph
    ponos::ivec2 bearing;     //!< offset from baseline to left/top of glyph
    GLuint advance;           //!< offset to advance to next glyph
  };
  FontTexture();
  ~FontTexture();
  /// \param c character ascii code
  /// \param s true size
  /// \param a glyph texture attributes
  /// \param p glyph texture parameters
  /// \param bearing offset from baseline to left/top of glyph
  /// \param advance offset to advance to next glyph
  void addCharacter(GLubyte c, ponos::ivec2 s, TextureAttributes a,
                    TextureParameters p, ponos::ivec2 bearing, GLuint advance);
  /// \param c character ascii code
  /// \return character object
  const Character &operator[](GLubyte c) const;
 private:
  std::map<GLchar, Character> characters;
};

} // aergia namespace

#endif //AERGIA_FONT_TEXTURE_H
