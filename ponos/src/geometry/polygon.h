#ifndef PONOS_GEOMETRY_POLYGON_H
#define PONOS_GEOMETRY_POLYGON_H

#include "geometry/point.h"
#include "geometry/shape.h"

#include <vector>

namespace ponos {

	class Polygon : public Shape {
  	public:
	 		Polygon() {}
			Polygon(std::vector<Point2> v)
				: vertices(v) {}
			virtual ~Polygon() {}

			std::vector<Point2> vertices;
	};

} // ponos namespace

#endif // PONOS_GEOMETRY_POLYGON_H

