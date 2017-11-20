#ifndef AERGIA_UI_TRACKBALL_H
#define AERGIA_UI_TRACKBALL_H

#include <ponos/ponos.h>

namespace aergia {

class Trackball {
public:
  Trackball() { radius = 5.f; }

  ponos::Point3 center;
  ponos::Transform transform;

  float radius;
  float angle_;
};

} // aergia namespace

#endif // AERGIA_UI_TRACKBALL_H
