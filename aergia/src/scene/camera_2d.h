#ifndef AERGIA_SCENE_CAMERA_2D_H
#define AERGIA_SCENE_CAMERA_2D_H

#include <ponos.h>

#include "scene/camera.h"

namespace aergia {

class Camera2D : public CameraInterface {
public:
  Camera2D();

  void look() override;
  void resize(float w, float h) override;
  void setZoom(float z);
  void setPos(ponos::vec2 p);
  void update();
  ponos::Transform getTransform() const override;
  ponos::Ray3 pickRay(ponos::Point2 p) const override;

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
