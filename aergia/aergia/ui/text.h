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

#ifndef AERGIA_UI_TEXT_H
#define AERGIA_UI_TEXT_H

#ifdef FREETYPE_INCLUDED
#include <ft2build.h>
#include FT_FREETYPE_H
#endif
#include <map>

#include <aergia/io/graphics_display.h>
#include <aergia/io/texture.h>
#include <aergia/scene/quad.h>
#include <aergia/utils/open_gl.h>

namespace aergia {

/** Draws texts on the screen. */
class Text {
public:
  /**
   * \brief Constructor
   * \param font file name.
   */
  Text(const char *font);
  /** \brief draws text on screen from a screen position
   * \param s text
   * \param x pixel position (screen coordinates)
   * \param y pixel position (screen coordinates)
   * \param scale
   * \param c color
   */
  void render(std::string s, GLfloat x, GLfloat y, GLfloat scale,
              aergia::Color c);
  /** \brief draws text on screen from a screen position
   * \param s text
   * \param p pixel position (in norm dev coordinates)
   * \param scale
   * \param c color
   */
  void render(std::string s, const ponos::Point3 &p, GLfloat scale,
              aergia::Color c);

private:
  struct Character {
    aergia::Texture *texture; //!< the glyph texture
    ponos::ivec2 size;        //!< size of glyph
    ponos::ivec2 bearing;     //!< offset from baseline to left/top of glyph
    GLuint advance;           //!< offset to advance to next glyph
  };
#ifdef FREETYPE_INCLUDED
  FT_Library ft;
  FT_Face ftFace;
#endif
  std::map<GLchar, Character> characters;
  Quad quad;
};

} // aergia namespace

#endif // AERGIA_UI_TEXT_H
