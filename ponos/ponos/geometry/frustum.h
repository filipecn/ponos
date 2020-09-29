#ifndef PONOS_GEOMETRY_FRUSTUM_H
#define PONOS_GEOMETRY_FRUSTUM_H

#include <ponos/geometry/plane.h>
#include <ponos/geometry/point.h>
#include <ponos/geometry/transform.h>
#include <ponos/geometry/vector.h>

namespace ponos {

class Frustum {
public:
  Frustum() = default;

  // http://gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf
  // it t is the projection matrix, then frustum is in view space
  // it t is the M * V * P matrix, then frustum is in model space
  void set(Transform t) {
    left.normal.x = t.matrix()[3][0] + t.matrix()[0][0];
    left.normal.y = t.matrix()[3][1] + t.matrix()[0][1];
    left.normal.z = t.matrix()[3][2] + t.matrix()[0][2];
    left.offset = t.matrix()[3][3] + t.matrix()[0][3];

    right.normal.x = t.matrix()[3][0] - t.matrix()[0][0];
    right.normal.y = t.matrix()[3][1] - t.matrix()[0][1];
    right.normal.z = t.matrix()[3][2] - t.matrix()[0][2];
    right.offset = t.matrix()[3][3] - t.matrix()[0][3];

    bottom.normal.x = t.matrix()[3][0] + t.matrix()[1][0];
    bottom.normal.y = t.matrix()[3][1] + t.matrix()[1][1];
    bottom.normal.z = t.matrix()[3][2] + t.matrix()[1][2];
    bottom.offset = t.matrix()[3][3] + t.matrix()[1][3];

    top.normal.x = t.matrix()[3][0] - t.matrix()[1][0];
    top.normal.y = t.matrix()[3][1] - t.matrix()[1][1];
    top.normal.z = t.matrix()[3][2] - t.matrix()[1][2];
    top.offset = t.matrix()[3][3] - t.matrix()[1][3];

    near_.normal.x = t.matrix()[3][0] + t.matrix()[2][0];
    near_.normal.y = t.matrix()[3][1] + t.matrix()[2][1];
    near_.normal.z = t.matrix()[3][2] + t.matrix()[2][2];
    near_.offset = t.matrix()[3][3] + t.matrix()[2][3];

    far_.normal.x = t.matrix()[3][0] - t.matrix()[2][0];
    far_.normal.y = t.matrix()[3][1] - t.matrix()[2][1];
    far_.normal.z = t.matrix()[3][2] - t.matrix()[2][2];
    far_.offset = t.matrix()[3][3] - t.matrix()[2][3];
  }
  [[nodiscard]] bool isInside(const ponos::point3 &p) const {
    return !near_.onNormalSide(p) && !far_.onNormalSide(p) &&
        !left.onNormalSide(p) && !right.onNormalSide(p) &&
        !bottom.onNormalSide(p) && !top.onNormalSide(p);
  }

  Plane near_;
  Plane far_;
  Plane left;
  Plane right;
  Plane bottom;
  Plane top;
};

} // namespace ponos

#endif
