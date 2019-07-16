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

#ifndef CIRCE_SCENE_CAMERA_H
#define CIRCE_SCENE_CAMERA_H

#include <ponos/ponos.h>

#include "camera_projection.h"
#include <circe/utils/open_gl.h>

namespace circe {

class CameraInterface {
public:
  typedef CameraInterface CameraType;
  /// \param p projection type
  explicit CameraInterface(CameraProjection *p = nullptr)
      : zoom(1.f), projection(p) {}
  virtual ~CameraInterface() = default;
  /// \param p normalized mouse position
  /// \return camera's ray in world space
  virtual ponos::Ray3 pickRay(ponos::point2 p) const {
    ponos::point3 O = ponos::inverse(model * view)(
        ponos::inverse(projection->transform) * ponos::point3(p.x, p.y, 0.f));
    ponos::point3 D = ponos::inverse(model * view)(
        ponos::inverse(projection->transform) * ponos::point3(p.x, p.y, 1.f));
    return ponos::Ray3(O, D - O);
  }
  /// multiplies MVP matrix to OpenGL pipeline
  virtual void look() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glApplyTransform(projection->transform);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glApplyTransform(view);
    glApplyTransform(model);
  }
  /// \param w width in pixels
  /// \param h height in pixels
  virtual void resize(float w, float h) {
    projection->ratio = w / h;
    projection->clipSize = ponos::vec2(zoom, zoom);
    if (w < h)
      projection->clipSize.y = projection->clipSize.x / projection->ratio;
    else
      projection->clipSize.x = projection->clipSize.y * projection->ratio;
    projection->update();
    update();
  }
  /// \return MVP transform
  virtual ponos::Transform getTransform() const {
    return projection->transform * view * model;
    return model * view * projection->transform;
  }
  /// \return camera projection const ptr
  const CameraProjection *getCameraProjection() const {
    return projection.get();
  }
  /// \return projection transform
  ponos::Transform getProjectionTransform() const {
    return projection->transform;
  }
  /// \return model transform
  virtual ponos::Matrix3x3<real_t> getNormalMatrix() const { return normal; }
  virtual ponos::Transform getModelTransform() const { return model; }
  virtual ponos::Transform getViewTransform() const { return view; }
  /// \param p normalized mouse position
  /// \return line object in world space
  virtual ponos::Line viewLineFromWindow(ponos::point2 p) const {
    ponos::point3 O = ponos::inverse(model * view)(
        ponos::inverse(projection->transform) * ponos::point3(p.x, p.y, 0.f));
    ponos::point3 D = ponos::inverse(model * view)(
        ponos::inverse(projection->transform) * ponos::point3(p.x, p.y, 1.f));
    return {O, D - O};
  }
  /// Applies **t** to camera transform (usually model transformation)
  /// \param t transform
  virtual void applyTransform(const ponos::Transform &t) {
    pos = target + t(pos - target) + t.getTranslate();
    target += t.getTranslate();
    up = t(up);
    // model = model * t;
    update();
  }
  /// \param p point in world space
  /// \return plane containing **p**, perpendicular to camera's view axis
  virtual ponos::Plane viewPlane(ponos::point3 p) const {
    ponos::vec3 n = pos - p;
    if (fabs(n.length()) < 1e-8)
      n = ponos::vec3(0, 0, 0);
    else
      n = ponos::normalize(n);
    return {ponos::normal3(n), ponos::dot(n, ponos::vec3(p.x, p.y, p.z))};
  }
  /// \return up vector
  virtual ponos::vec3 getUpVector() const { return up; };
  /// \return eye position
  virtual ponos::point3 getPosition() const { return pos; };
  /// \return  target position
  virtual ponos::point3 getTarget() const { return target; }
  /// \param p target position
  virtual void setTarget(ponos::point3 p) {
    target = p;
    update();
  }
  /// \param p eye position
  virtual void setPosition(ponos::point3 p) {
    pos = p;
    update();
  }
  /// \param z
  virtual void setZoom(float z) {
    zoom = z;
    update();
  }
  /// \param f
  virtual void setFar(float f) {
    projection->zfar = f;
    projection->update();
  }
  /// \param n
  virtual void setNear(float n) {
    projection->znear = n;
    projection->update();
  }
  virtual float getNear() const { return projection->znear; }
  virtual float getFar() const { return projection->zfar; }
  virtual const ponos::Frustum &getFrustum() const { return frustum; }
  ///
  virtual void update() = 0;

protected:
  float zoom;
  ponos::vec3 up;
  ponos::point3 pos;
  ponos::point3 target;
  ponos::Transform view;
  ponos::Transform model;
  ponos::Matrix3x3<real_t> normal;
  ponos::Frustum frustum;
  std::shared_ptr<CameraProjection> projection;
};

} // namespace circe

#endif // CIRCE_SCENE_CAMERA_H
