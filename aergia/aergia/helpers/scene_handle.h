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

#ifndef AERGIA_HELPERS_SCENE_HANDLE_H
#define AERGIA_HELPERS_SCENE_HANDLE_H

#include <aergia/colors/color_palette.h>
#include <aergia/scene/scene_object.h>
#include <functional>

namespace aergia {

template <typename T> class CircleHandle : public SceneObject {
public:
  CircleHandle() {
    fillColor = selectedColor = COLOR_BLUE;
    fillColor.a = 0.1f;
    circle.c = ponos::Point2();
    circle.r = 0.1f;
  }
  void draw() const override {
    glColor(fillColor);
    if (this->selected)
      glColor(selectedColor);
    // draw_circle(circle, &this->transform);
  }

  bool intersect(const ponos::Ray3 &r, float *t = nullptr) override {
    return ponos::distance2(circle.c, ponos::Point2(r.o.x, r.o.y)) <=
           SQR(circle.r);
  }

  ponos::Circle circle;
  Color fillColor;
  Color selectedColor;
  std::function<void(T *)> updateCallback;

private:
  T *data;
};

} // aergia namespace

#endif // AERGIA_HELPERS_SCENE_HANDLE_H
