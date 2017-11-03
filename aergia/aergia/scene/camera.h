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

#ifndef AERGIA_SCENE_CAMERA_H
#define AERGIA_SCENE_CAMERA_H

#include <ponos/ponos.h>

#include "camera_projection.h"

namespace aergia {

class CameraInterface {
public:
  typedef CameraInterface CameraType;
  CameraInterface() {}
  virtual ~CameraInterface() {}

  virtual ponos::Ray3 pickRay(ponos::Point2 p) const = 0;
  virtual void look() = 0;
  virtual void resize(float w, float h) = 0;
  virtual ponos::Transform getTransform() const = 0;
  virtual ponos::Line viewLineFromWindow(ponos::Point2 p) const = 0;
};

class Camera : public CameraInterface {
public:
  typedef Camera CameraType;
  Camera(CameraProjection *_projection = nullptr);

  friend class CameraModel;

  void look() override;
  void resize(float w, float h);
  void setZoom(float z);
  ponos::Point3 getPos() const { return pos; }
  void setPos(ponos::Point3 p);
  ponos::Point3 getTarget() const { return target; }
  void setTarget(ponos::Point3 p);
  void setUp(const ponos::vec3 &u);
  void setFov(float f);
  void setFar(float f);
  void setNear(float n);
  float getNear() const { return projection->znear; }
  float getFar() const { return projection->zfar; }
  void update();
  ponos::Transform getTransform() const override;
  ponos::Point3 viewPointOnWorldCoord() const;
  // p must be in norm dev coord (windowCoordToNormDevCoord)
  ponos::Line viewLineFromWindow(ponos::Point2 p) const override;
  ponos::Ray3 pickRay(ponos::Point2 p) const override;
  ponos::Plane viewPlane(ponos::Point3 p) const;

private:
  float zoom;
  ponos::Point3 pos, target;
  ponos::vec3 up;
  ponos::vec2 display;
  ponos::Transform view;
  ponos::Transform model;
  ponos::Frustum frustum;
  std::shared_ptr<CameraProjection> projection;
};

} // aergia namespace

#endif // AERGIA_SCENE_CAMERA_H
