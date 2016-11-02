#ifndef PONOS_GEOMETRY_POLYGON_H
#define PONOS_GEOMETRY_POLYGON_H

#include "geometry/bbox.h"
#include "geometry/point.h"
#include "geometry/shape.h"

#include <vector>

namespace ponos {

	class Polygon : public Shape {
  	public:
	 		Polygon() {}
			Polygon(std::vector<Point2> v)
				: vertices(v) {}
			virtual ~Polygon() { vertices.clear(); }

			std::vector<Point2> vertices;
	};

  inline BBox2D compute_bbox(const Polygon& po) {
    BBox2D b;
    for (auto p : po.vertices)
      b = make_union(b, p);
    return b;
  }

} // ponos namespace

#endif // PONOS_GEOMETRY_POLYGON_H
