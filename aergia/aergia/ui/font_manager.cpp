// Created by filipecn on 3/28/18.
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

#include "font_manager.h"

namespace aergia {

FontManager FontManager::instance_;

FontManager &FontManager::instance() {
  // TODO it has to be called after opengl is initialized! (this code shouldn't be here!
#ifdef ASSET_PATH
  if (!instance_.initialized_) {
    std::string arial(ASSET_PATH);
    arial += "/arial.ttf";
    instance_.loadFromFile(arial.c_str());
  }
#endif
  return instance_;
}

int FontManager::loadFromFile(const char *filename) {
  init();
  FontTexture font;
#ifdef FREETYPE_INCLUDED
  FT_Face ftFace;
  ASSERT_MESSAGE(!FT_New_Face(ft, filename, 0, &ftFace),
                 "ERROR::FREETYPE: Failed to load font");
  FT_Set_Pixel_Sizes(ftFace, 0, 48);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // Disable byte-alignment restriction
  for (GLubyte c = 0; c < 128; c++) {
    // Load character glyph
    if (FT_Load_Char(ftFace, c, FT_LOAD_RENDER)) {
      std::cerr << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
      continue;
    }
    aergia::TextureAttributes attributes;
    attributes.internalFormat = GL_R8;
    attributes.target = GL_TEXTURE_2D;
    attributes.width = ftFace->glyph->bitmap.width;
    attributes.height = ftFace->glyph->bitmap.rows;
    attributes.format = GL_RED;
    attributes.type = GL_UNSIGNED_BYTE;
    attributes.data = ftFace->glyph->bitmap.buffer;
    aergia::TextureParameters parameters;
    // Now store character for later use
    font.addCharacter(
        c,
        ponos::ivec2(ftFace->glyph->bitmap.width, ftFace->glyph->bitmap.rows),
        attributes, parameters,
        ponos::ivec2(ftFace->glyph->bitmap_left, ftFace->glyph->bitmap_top),
        static_cast<GLuint>(ftFace->glyph->advance.x));
  }
  FT_Done_Face(ftFace);
  fonts_.emplace_back(font);
  return static_cast<int>(fonts_.size() - 1);
#endif
  UNUSED_VARIABLE(filename);
  return -1;
}

FontManager::FontManager() : initialized_(false) {
}

void FontManager::init() {
  if (initialized_)
    return;
#ifdef FREETYPE_INCLUDED
  ASSERT_MESSAGE(!FT_Init_FreeType(&ft),
                 "ERROR::FREETYPE: Could not init FreeType Library");
#endif
  initialized_ = true;
}

FontManager::~FontManager() {
#ifdef FREETYPE_INCLUDED
  FT_Done_FreeType(ft);
#endif
}

const FontTexture &FontManager::fontTexture(size_t id) {
  init();
  return fonts_[id];
}

} // aergia namespace
