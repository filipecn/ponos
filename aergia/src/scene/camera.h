#ifndef AERGIA_SCENE_CAMERA_H
#define AERGIA_SCENE_CAMERA_H

#include <ponos.h>

#include "camera_projection.h"

namespace aergia {

class CameraInterface {
public:
  CameraInterface() {}
  virtual ~CameraInterface() {}

  virtual ponos::Ray3 pickRay(ponos::Point2 p) const = 0;
  virtual void look() = 0;
  virtual void resize(float w, float h) = 0;
  virtual ponos::Transform getTransform() const = 0;
};

class Camera : public CameraInterface {
public:
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
  ponos::Line viewLineFromWindow(ponos::Point2 p) const;
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
