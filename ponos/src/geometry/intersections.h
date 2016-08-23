#ifndef PONOS_GEOMETRY_INTERSECTIONS_H
#define PONOS_GEOMETRY_INTERSECTIONS_H

#include "geometry/bbox.h"
#include "geometry/line.h"
#include "geometry/plane.h"
#include "geometry/ray.h"
#include "geometry/sphere.h"
#include "geometry/utils.h"
#include "geometry/vector.h"

namespace ponos {

	bool plane_line_intersection(const Plane pl, const Line l, Point3& p);
	bool sphere_line_intersection(const Sphere s, const Line l, Point3& p1, Point3& p2);
	bool bbox_ray_intersection(const BBox& box, const Ray3& ray, float& hit1, float& hit2);
	bool bbox_ray_intersection(const BBox& box, const Ray3& ray, float& hit1);
	bool triangle_ray_intersection(const Point3& p1, const Point3& p2, const Point3& p3, const Ray3& ray, float *tHit, float *b1 = nullptr, float *b2 = nullptr);
} // ponos namespace

#endif
