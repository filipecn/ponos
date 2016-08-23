#include "geometry/intersections.h"

#include <utility>

namespace ponos {

	bool plane_line_intersection(const Plane pl, const Line l, Point3& p) {
		vec3 nvector = vec3(pl.normal.x, pl.normal.y, pl.normal.z);
		float k = dot(nvector, l.direction());
		if(IS_ZERO(k))
			return false;
		float r = (pl.offset - dot(nvector, vec3(l.a.x, l.a.y, l.a.z))) / k;
		p = l(r);
		return true;
	}

	bool sphere_line_intersection(const Sphere s, const Line l, Point3& p1, Point3& p2) {
			float a = dot(l.direction(), l.direction());
			float b = 2.f * dot(l.direction(), l.a - s.c);
			float c = dot(l.a - s.c, l.a - s.c) - s.r * s.r;

			float delta = b * b - 4 * a * c;
			if(delta < 0)
				return false;

			float d = sqrt(delta);

			p1 = l((-b + d) / (2.0 * a));
			p2 = l((-b - d) / (2.0 * a));
			return true;
	}

	bool bbox_ray_intersection(const BBox& box, const Ray3& ray, float& hit0, float& hit1) {
		float t0 = 0.f, t1 = INFINITY;
		for(int i = 0; i < 3; i++) {
			float invRayDir = 1.f / ray.d[i];
			float tNear = (box.pMin[i] - ray.o[i]) * invRayDir;
			float tFar  = (box.pMax[i] - ray.o[i]) * invRayDir;
			if(tNear > tFar)
				std::swap(tNear, tFar);
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar  > t1 ? tFar  : t1;
			if(t0 > t1)
				return false;
		}
		hit0 = t0;
		hit1 = t1;
		return true;
	}

	bool bbox_ray_intersection(const BBox& box, const Ray3& ray, float& hit0) {
		return bbox_ray_intersection(box, ray, hit0);
	}

	bool triangle_ray_intersection(const Point3& p1, const Point3& p2, const Point3& p3, const Ray3& ray, float *tHit, float* b1, float* b2) {
		// Compute s1
		vec3 e1 = p2 - p1;
		vec3 e2 = p3 - p1;
		vec3 s1 = cross(ray.d, e2);
		float divisor = dot(s1, e1);
		if(divisor == 0.f)
			return false;
		float invDivisor = 1.f / divisor;
		// Compute first barycentric coordinate
		vec3 d = ray.o - p1;
		float b1_ = dot(d, s1) * invDivisor;
		if(b1_ < 0.f || b1_ > 1.f)
			return false;
		// Compute second barycentric coordinate
		vec3 s2 = cross(d, e1);
		float b2_ = dot(ray.d, s2) * invDivisor;
		if(b2_ < 0. || b1_ + b2_ > 1.)
			return false;
		// Compute t to intersection point
		float t = dot(e2, s2) * invDivisor;
		if(t < 0.)
			return false;
		*tHit = t;
		if(b1)
			*b1 = b1_;
		if(b2)
			*b2 = b2_;
		return true;
	}

} // ponos namespace
