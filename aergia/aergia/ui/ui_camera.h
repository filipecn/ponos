// Created by filipecn on 3/2/18.
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

#ifndef PONOS_UI_CAMERA_H
#define PONOS_UI_CAMERA_H

#include <aergia/scene/camera.h>
#include <aergia/ui/trackball_interface.h>

namespace aergia {

/// Represents an user controllable camera through a trackball
class UserCamera : public CameraInterface {
public:
  UserCamera();
  /// process mouse button event
  /// \param action event type
  /// \param button button code
  /// \param p normalized mouse position
  void mouseButton(int action, int button, ponos::Point2 p);
  /// process mouse move event
  /// \param p normalized mouse position
  void mouseMove(ponos::Point2 p);
  /// process mouse wheel event
  /// \param p normalized mouse position
  /// \param d scroll vector
  virtual void mouseScroll(ponos::Point2 p, ponos::vec2 d);
  TrackballInterface trackball;
};

class UserCamera2D : public UserCamera {
public:
  UserCamera2D();
  void mouseScroll(ponos::Point2 p, ponos::vec2 d) override;
  void fit(const ponos::BBox2D &b, float delta = 1.f);
  void update() override;
};

class UserCamera3D : public UserCamera {
public:
  explicit UserCamera3D();
  void setUp(const ponos::vec3 &u);
  void setFov(float f);
  void update() override;
};

} // aergia namespace

#endif //PONOS_UI_CAMERA_H
