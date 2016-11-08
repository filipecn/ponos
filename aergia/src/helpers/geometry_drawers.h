#ifndef AERGIA_HELPERS_GEOMETRY_DRAWERS_H
#define AERGIA_HELPERS_GEOMETRY_DRAWERS_H

#include "utils/open_gl.h"

#include <ponos.h>

namespace aergia {

	void draw_segment(ponos::Segment3 segment);

	void draw_circle(ponos::Circle circle);

	void draw_sphere(ponos::Sphere sphere);

	void draw_polygon(const ponos::Polygon &polygon, const ponos::Transform2D* transform = nullptr);

} // aergia namespace

#endif // AERGIA_HELPERS_GEOMETRY_DRAWERS_H
