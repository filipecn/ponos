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

#ifndef CIRCE_UI_MODIFIER_CURSOR_H
#define CIRCE_UI_MODIFIER_CURSOR_H

#include <circe/colors/color_palette.h>
#include <circe/scene/scene_object.h>
#include <functional>

namespace circe {

class ModifierCursor : public SceneObject {
public:
  ModifierCursor() {
    this->active = true;
    dragging = false;
  }
  virtual ~ModifierCursor() {}

  void mouse(CameraInterface &camera, ponos::point2 p) override {
    UNUSED_VARIABLE(camera);
    ponos::point3 P = inverse(glGetMVPTransform())(p);
    last = position;
    position = ponos::point2(P.x, P.y);
    static bool first = true;
    if (first)
      last = position, first = false;
    mouseMove();
  }

  void button(CameraInterface &camera, ponos::point2 p, int button,
              int action) override {
    UNUSED_VARIABLE(p);
    UNUSED_VARIABLE(camera);
    if (action == GLFW_RELEASE) {
      dragging = false;
    } else {
      dragging = true;
      start = position;
    }
    mouseButton(button, action);
  }

  bool intersect(const ponos::Ray3 &r, float *t = nullptr) override {
    UNUSED_VARIABLE(r);
    *t = 0.0000001;
    return true;
  }

  virtual void mouseMove() {}
  virtual void mouseButton(int b, int a) {
    UNUSED_VARIABLE(a);
    UNUSED_VARIABLE(b);
  }

  bool dragging;
  ponos::point2 position;
  ponos::point2 start;
  ponos::point2 last;
};

class CircleCursor : public ModifierCursor {
public:
  CircleCursor(const ponos::Circle &c,
               const std::function<void(const CircleCursor &, ponos::vec2)> &f,
               Color fc = COLOR_BLACK, Color ac = COLOR_RED)
      : mouseCallback(f), fillColor(fc), activeColor(ac), circle(c) {
    fillColor.a = 0.1;
    activeColor.a = 0.3;
  }
  virtual ~CircleCursor() {}

  void draw(const CameraInterface *camera,
            ponos::Transform transform) override {
    ponos::Circle c = circle;
    if (this->dragging)
      glColor(activeColor);
    else
      glColor(fillColor);
    draw_circle(c);
  }

  void mouseMove() override {
    circle.c = this->position;
    if (mouseCallback)
      mouseCallback(*this, position - last);
  }

  void mouseButton(int b, int a) override {
    if (buttonCallback)
      buttonCallback(*this, b, a);
  }

  std::function<void(const CircleCursor &, ponos::vec2)> mouseCallback;
  std::function<void(const CircleCursor &, int b, int a)> buttonCallback;
  Color fillColor;
  Color activeColor;
  ponos::Circle circle;
};

} // namespace circe

#endif // CIRCE_UI_MODIFIER_CURSOR_H
