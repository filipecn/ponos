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

#ifndef AERGIA_TEXT_MANAGER_H
#define AERGIA_TEXT_MANAGER_H

#ifdef FREETYPE_INCLUDED
#include <ft2build.h>
#include FT_FREETYPE_H
#endif

#include <aergia/ui/text_renderer.h>
#include <aergia/io/font_texture.h>

namespace aergia {

/// Manages font texts
class FontManager {
public:
  static FontManager &instance();
  virtual ~FontManager();
  FontManager(FontManager const &) = delete;
  void operator=(FontManager const &) = delete;
  /// \param filename .ttf full path
  /// \return font id
  int loadFromFile(const char *filename);
  /// \param id font id
  /// \return font texture
  const FontTexture& fontTexture(size_t id = 0);
private:
  FontManager();
  void init();
  static FontManager instance_;
  bool initialized_;
  std::vector<FontTexture> fonts_; //!< fonts textures
#ifdef FREETYPE_INCLUDED
  FT_Library ft;
#endif
};

} // aergia namespace

#endif //AERGIA_TEXT_MANAGER_H