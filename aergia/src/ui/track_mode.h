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

#ifndef AERGIA_UI_TRACK_MODE_H
#define AERGIA_UI_TRACK_MODE_H

#include "helpers/geometry_drawers.h"
#include "scene/camera.h"
#include "ui/trackball.h"

#include <ponos.h>
#include <iostream>

namespace aergia {

/** \brief Defines how a trackball is manipulated. */
class TrackMode {
public:
  TrackMode() : dragging_(false) {}
  virtual ~TrackMode() {}

  virtual void draw(const Trackball &tb) {}

  // CONTROL
  virtual void start(Trackball &tb, const CameraInterface &camera,
                     ponos::Point2 p) {
    start_ = p;
    dragging_ = true;
  }
  virtual void update(Trackball &tb, Camera &camera, ponos::Point2 p) = 0;
  virtual void stop(Trackball &tb, Camera &camera, ponos::Point2 p) {
    dragging_ = false;
    tb.transform = ponos::Transform();
  }
  virtual void update(Trackball &tb, CameraInterface &camera, ponos::vec2 d) {}

protected:
  ponos::Point3 hitViewPlane(Trackball &tb, Camera &camera, ponos::Point2 p) {
    ponos::Line l = camera.viewLineFromWindow(p);
    ponos::Plane vp = camera.viewPlane(tb.center);
    ponos::Point3 hp;
    ponos::plane_line_intersection(vp, l, hp);
    return hp;
  }

  bool dragging_;
  ponos::Point2 start_;
};

class ScaleMode : public TrackMode {
public:
  ScaleMode() : TrackMode() {}
  ~ScaleMode() {}
  void update(Trackball &tb, Camera &camera, ponos::Point2 p) override {}
  void update(Trackball &tb, CameraInterface &camera, ponos::vec2 d) override {
    float scale = (d.y > 0.f) ? 1.1f : 0.9f;
    tb.transform = tb.transform * ponos::scale(scale, scale, scale);
  }
};

class PanMode : public TrackMode {
public:
  PanMode() : TrackMode() {}
  ~PanMode() {}

  void draw(Trackball tb) {}

  void update(Trackball &tb, Camera &camera, ponos::Point2 p) override {
    if (!dragging_)
      return;
    ponos::Point3 a = hitViewPlane(tb, camera, start_);
    ponos::Point3 b = hitViewPlane(tb, camera, p);
    tb.transform = ponos::translate(b - a);
    start_ = p;
  }
};

class ZMode : public TrackMode {
public:
  ZMode() : TrackMode() {}
  ~ZMode() {}

  void update(Trackball &tb, Camera &camera, ponos::Point2 p) override {
    if (!dragging_)
      return;
    ponos::Point3 a = hitViewPlane(tb, camera, start_);
    ponos::Point3 b = hitViewPlane(tb, camera, p);
    ponos::vec3 dir = ponos::normalize(camera.getTarget() - camera.getPos());
    if (p.y - start_.y < 0.f)
      dir *= -1.f;
    tb.transform = ponos::translate(dir * ponos::distance(a, b));
    start_ = p;
  }
};

class RotateMode : public TrackMode {
public:
  RotateMode() : TrackMode() {}
  ~RotateMode() {}
  void draw(const Trackball &tb) override {
    ponos::Sphere s(tb.center, tb.radius * 2.f);
    glColor4f(0, 0, 0, 0.5);
    draw_sphere(s);
  }
  void update(Trackball &tb, Camera &camera, ponos::Point2 p) override {
    if (!dragging_ || p == start_)
      return;
    // std::cout << start_;
    // std::cout << p;
    ponos::Point3 a = hitSpherePlane(tb, camera, start_);
    ponos::Point3 b = hitSpherePlane(tb, camera, p);

    ponos::vec3 axis =
        ponos::normalize(ponos::cross((a - tb.center), (b - tb.center)));
    // std::cout << "a " << a;
    // std::cout << "b " << b;
    // std::cout << "axis " << axis;

    float phi = ponos::distance(a, b) / tb.radius;
    // std::cout << phi << std::endl;
    /*		vec3 newAxis = glm::normalize(glm::inverse(
                                            glm::mat3(
                                                    glm::toMat4(tb.transform.r)
                                                    *
                                                    modelviewMatrix()
                                                    ))*axis);*/
    // cout << newAxis << endl;
    // tb.transform.r = glm::angleAxis(glm::degrees(phi)*0.4f, newAxis);
    tb.transform = ponos::rotate(TO_DEGREES(phi) * 0.4f, axis);

    start_ = p;
  }

private:
  ponos::Point3 hitSpherePlane(Trackball &tb, Camera &camera, ponos::Point2 p) {
    ponos::Line l = camera.viewLineFromWindow(p);
    ponos::Plane vp = camera.viewPlane(tb.center);

    ponos::Sphere s(tb.center, tb.radius);

    ponos::Point3 hp;
    ponos::plane_line_intersection(vp, l, hp);

    ponos::Point3 hs, hs1, hs2;
    bool resSp = sphere_line_intersection(s, l, hs1, hs2);
    if (resSp) {
      if (ponos::distance(camera.getPos(), hs1) <
          ponos::distance(camera.getPos(), hs2))
        hs = hs1;
      else
        hs = hs2;
      return hs;
    }
    ponos::Point3 hh;
    bool resHp = hitHyper(tb, camera.getPos(), vp, hp, hh);
    if ((!resSp && !resHp))
      return l.closestPoint(tb.center);
    if ((resSp && !resHp))
      return hs;
    if ((!resSp && resHp))
      return hh;
    float angleDeg = TO_DEGREES(
        asin(ponos::dot(ponos::normalize(camera.getPos() - tb.center),
                        ponos::normalize(hs - tb.center))));
    if (angleDeg < 45)
      return hs;
    else
      return hh;
  }
  bool hitHyper(Trackball tb, ponos::Point3 viewpoint, ponos::Plane vp,
                ponos::Point3 hitplane, ponos::Point3 &hit) {
    float hitplaney = ponos::distance(tb.center, hitplane);
    float viewpointx = ponos::distance(tb.center, viewpoint);

    float a = hitplaney / viewpointx;
    float b = -hitplaney;
    float c = (float)((tb.radius * tb.radius) / 2.f);
    float delta = b * b - 4 * a * c;
    float x1, xval, yval;

    if (delta > 0) {
      x1 = (-b - sqrtf(delta)) / (2.f * a);
      xval = x1;
      yval = c / xval;
    } else
      return false;
    ponos::vec3 dirRadial = ponos::normalize(hitplane - tb.center);
    ponos::vec3 dirView = ponos::normalize(ponos::vec3(vp.normal));
    hit = tb.center + dirRadial * yval + dirView * xval;
    return true;
  }
};

} // aergia namespace

#endif // AERGIA_UI_TRACK_MODE_H
