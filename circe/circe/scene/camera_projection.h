#ifndef CIRCE_SCENE_CAMERA_PROJECTION_H
#define CIRCE_SCENE_CAMERA_PROJECTION_H

#include <ponos/ponos.h>

namespace circe {

class CameraProjection {
public:
  CameraProjection() {
    ratio = 1.f;
    znear = 0.01f;
    zfar = 1000.f;
  }
  virtual ~CameraProjection() = default;

  virtual void update() = 0;

  float ratio;
  float znear;
  float zfar;
  ponos::vec2 clipSize;
  ponos::Transform transform;
};

class PerspectiveProjection : public CameraProjection {
public:
  explicit PerspectiveProjection(float _fov = 45.f) : fov(_fov) {}
  void update() override {
    this->transform =
        ponos::perspective(fov, this->ratio, this->znear, this->zfar);
  }

  float fov;
};

class OrthographicProjection : public CameraProjection {
public:
  OrthographicProjection() {
    _region.pMin.x = _region.pMin.y = this->znear = -1.f;
    _region.pMax.x = _region.pMax.y = this->zfar = 1.f;
  }
  void zoom(float z) { _region = ponos::scale(z, z)(_region); }
  void set(float left, float right, float bottom, float top) {
    _region.pMin.x = left;
    _region.pMin.y = bottom;
    _region.pMax.x = right;
    _region.pMax.y = top;
    update();
  }
  void update() override {
    this->transform =
        ponos::ortho(_region.pMin.x, _region.pMax.x, _region.pMin.y,
                     _region.pMax.y, this->znear, this->zfar);
  }

private:
  ponos::BBox2D _region;
};

} // circe namespace

#endif // CIRCE_SCENE_CAMERA_PROJECTION_H
