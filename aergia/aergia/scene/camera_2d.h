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

#ifndef AERGIA_SCENE_CAMERA_2D_H
#define AERGIA_SCENE_CAMERA_2D_H

#include <aergia/scene/camera.h>

namespace aergia {

class Camera2D : public CameraInterface {
public:
  Camera2D();
  typedef Camera2D CameraType;
  void look() override;
  void resize(float w, float h) override;
  void fit(const ponos::BBox2D &b, float delta = 1.f);
  void setZoom(float z);
  void setPos(ponos::vec2 p);
  ponos::vec2 getPos() const { return pos; }
  ponos::Point3 getPosition() const override {
    return ponos::Point3(pos.x, pos.y, 0);
  }
  void setPosition(ponos::Point3 p) override;
  ponos::Point3 getTarget() const override {
    return ponos::Point3(pos.x, pos.y, 0);
  }
  void update();
  ponos::Transform getTransform() const override;
  ponos::Transform getModelTransform() const override;
  ponos::Ray3 pickRay(ponos::Point2 p) const override;
  ponos::Line viewLineFromWindow(ponos::Point2 p) const override;
  void applyTransform(const ponos::Transform &t) override;
  ponos::Plane viewPlane(ponos::Point3 p) const override;

private:
  float ratio;
  float zoom;
  ponos::vec2 pos;
  ponos::vec2 display;
  ponos::vec2 clipSize;
  ponos::Transform projection;
  ponos::Transform view;
  ponos::Transform model;
};

} // aergia namespace

#endif // AERGIA_SCENE_CAMERA_2D_H
