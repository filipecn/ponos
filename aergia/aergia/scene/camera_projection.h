#ifndef AERGIA_SCENE_CAMERA_PROJECTION_H
#define AERGIA_SCENE_CAMERA_PROJECTION_H

#include <ponos/ponos.h>

namespace aergia {

class CameraProjection {
public:
  CameraProjection() {}
  virtual ~CameraProjection() {}

  virtual void update() = 0;

  float ratio;
  float znear;
  float zfar;
  ponos::vec2 clipSize;
  ponos::Transform transform;
};

class PerspectiveProjection : public CameraProjection {
public:
  PerspectiveProjection(float _fov = 45.f) : fov(_fov) {}
  void update() override {
    this->transform =
        ponos::perspective(fov, this->ratio, this->znear, this->zfar);
  }

  float fov;
};

class OrthographicProjection : public CameraProjection {
public:
  OrthographicProjection() {}
  void update() override {
    this->transform =
        ponos::scale(1.f, 1.f, 1.f / (this->zfar - this->znear)) *
        ponos::translate(
            ponos::vec3(0.f, 0.f, -this->znear)); // ponos::ortho(-1, 1, -1, 1,
                                                  // this->znear, this->zfar);
  }
};

} // aergia namespace

#endif // AERGIA_SCENE_CAMERA_PROJECTION_H
