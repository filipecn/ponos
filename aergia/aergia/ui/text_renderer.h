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

#ifndef AERGIA_UI_TEXT_RENDERER_H
#define AERGIA_UI_TEXT_RENDERER_H

#include <map>

#include <aergia/io/graphics_display.h>
#include <aergia/io/texture.h>
#include <aergia/scene/quad.h>
#include <aergia/utils/open_gl.h>
#include <sstream>

namespace aergia {

/// Draws texts on the screen.
class TextRenderer {
public:
  /// \param s text size
  /// \param c text color
  /// \param id font id (from font manager)
  explicit TextRenderer(float scale = 1.f, Color c = COLOR_BLACK,
                        size_t id = 0);
  /// \brief draws text on screen from a screen position
  /// \param s text
  /// \param x pixel position (screen coordinates)
  /// \param y pixel position (screen coordinates)
  /// \param scale
  /// \param c color
  void render(std::string s, GLfloat x, GLfloat y, GLfloat scale,
              aergia::Color c);
  /// \brief draws text on screen from a screen position
  /// \param s text
  /// \param p pixel position (in world coordinates)
  /// \param scale
  /// \param c color
  void render(std::string s, const ponos::Point3 &p, GLfloat scale,
              aergia::Color c);
  /// \param p position (world coordinates)
  /// \return text renderer reference
  TextRenderer &at(const ponos::Point3 &p);
  /// \param p position (screen coordinates)
  /// \return text renderer reference
  TextRenderer &at(const ponos::Point2 &p);
  /// \param s scale
  /// \return text renderer reference
  TextRenderer &withScale(float s);
  /// \param c color
  /// \return text renderer reference
  TextRenderer &withColor(Color c);
  /// \param tr text renderer reference
  /// \return text renderer reference
  TextRenderer &operator<<(TextRenderer &tr);
  template<typename T> TextRenderer &operator<<(T t) {
    std::stringstream s;
    s << t;
    render(s.str(), dynamicPosition_.x, dynamicPosition_.y, dynamicScale_,
           dynamicColor_);
    return *this;
  }
  size_t fontId;   //!< font id (from font manager)
  float scale;     //!< text scale
  Color textColor; //!< text color
private:
  ponos::Point2 dynamicPosition_;
  float dynamicScale_;
  Color dynamicColor_;
  Quad quad_;
};

} // aergia namespace

#endif // AERGIA_UI_TEXT_H
