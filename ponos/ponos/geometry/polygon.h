#ifndef PONOS_GEOMETRY_POLYGON_H
#define PONOS_GEOMETRY_POLYGON_H

#include <ponos/geometry/bbox.h>
#include <ponos/geometry/point.h>
#include <ponos/geometry/shape.h>
#include <ponos/geometry/transform.h>

#include <vector>

namespace ponos {

class Polygon : public Shape {
public:
  Polygon() { this->type = ShapeType::POLYGON; }
  Polygon(std::vector<Point2> v) : vertices(v) {}
  /*
  Polygon(const Polygon& p) {
    vertices = p.vertices;
  }
  Polygon(Polygon&& p) noexcept {
    vertices = p.vertices;
  }
  Polygon& operator=(const Polygon& other) {
    Polygon tmp(other);
    *this = std::move(tmp);
    return *this;
  }*/

  Transform2D transform;
  std::vector<Point2> vertices;
};

inline BBox2D compute_bbox(const Polygon &po, const Transform2D *t = nullptr) {
  BBox2D b;
  for (auto p : po.vertices) {
    if (t != nullptr)
      b = make_union(b, (*t)(p));
    else
      b = make_union(b, p);
  }
  return b;
}

} // ponos namespace

#endif // PONOS_GEOMETRY_POLYGON_H
