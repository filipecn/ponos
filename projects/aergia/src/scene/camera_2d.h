#pragma once

#include <ponos.h>
using ponos::Transform;
using ponos::vec2;

namespace aergia {

  class Camera2D {
  public:
    Camera2D();

    void look();
    void resize(float w, float h);
    void setZoom(float z);
    void setPos(vec2 p);
    void update();

  private:
    float ratio;
    float zoom;
    vec2 pos;
    vec2 display;
    vec2 clipSize;
    Transform projection;
    Transform view;
    Transform model;
  };
} // aergia namespace
