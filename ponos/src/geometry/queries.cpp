#include "geometry/queries.h"

#include "geometry/matrix.h"
#include "geometry/numeric.h"

#include <utility>

namespace ponos {

bool bbox_bbox_intersection(const BBox2D &a, const BBox2D &b) {
  for (int i = 0; i < 2; i++)
    if (!((a.pMin[i] <= b.pMin[i] && a.pMax[i] >= b.pMin[i]) ||
          (b.pMin[i] <= a.pMin[i] && b.pMax[i] >= a.pMin[i])))
      return false;
  return true;
}

bool ray_segment_intersection(const Ray2 &R, const Segment2 &s, float *t) {
  vec2 dd = s.b - s.a;
  Ray3 r(Point3(R.o.x, R.o.y, 0.f), vec3(R.d.x, R.d.y, 0.f));
  Ray3 r2(Point3(s.a.x, s.a.y, 0.f), vec3(dd.x, dd.y, 0.f));
  float d = cross(r.d, r2.d).length2();
  if (d == 0.f)
    // TODO handle this case
    return false;
  float t1 = Matrix3x3(r2.o - r.o, r2.d, cross(r.d, r2.d)).determinant() / d;
  float t2 = Matrix3x3(r2.o - r.o, r.d, cross(r.d, r2.d)).determinant() / d;
  // TODO add user threshold
  float dist = (r(t1) - r2(t2)).length();
  if (t1 >= 0.f && t2 >= 0.f && t2 <= s.length() && dist < 0.1f) {
    if (t)
      *t = t1;
    return true;
  }
  return false;
}

bool ray_segment_intersection(const Ray3 &r, const Segment3 &s, float *t) {
  Ray3 r2(s.a, s.b - s.a);
  float d = cross(r.d, r2.d).length2();
  if (d == 0.f)
    // TODO handle this case
    return false;
  float t1 = Matrix3x3(r2.o - r.o, r2.d, cross(r.d, r2.d)).determinant() / d;
  float t2 = Matrix3x3(r2.o - r.o, r.d, cross(r.d, r2.d)).determinant() / d;
  // TODO add user threshold
  float dist = (r(t1) - r2(t2)).length();
  if (t1 >= 0.f && t2 >= 0.f && t2 <= s.length() && dist < 0.1f) {
    if (t)
      *t = t1;
    return true;
  }
  return false;
}

bool plane_line_intersection(const Plane pl, const Line l, Point3 &p) {
  vec3 nvector = vec3(pl.normal.x, pl.normal.y, pl.normal.z);
  float k = dot(nvector, l.direction());
  if (IS_ZERO(k))
    return false;
  float r = (pl.offset - dot(nvector, vec3(l.a.x, l.a.y, l.a.z))) / k;
  p = l(r);
  return true;
}

bool sphere_line_intersection(const Sphere s, const Line l, Point3 &p1,
                              Point3 &p2) {
  float a = dot(l.direction(), l.direction());
  float b = 2.f * dot(l.direction(), l.a - s.c);
  float c = dot(l.a - s.c, l.a - s.c) - s.r * s.r;

  float delta = b * b - 4 * a * c;
  if (delta < 0)
    return false;

  float d = sqrt(delta);

  p1 = l((-b + d) / (2.0 * a));
  p2 = l((-b - d) / (2.0 * a));
  return true;
}

bool sphere_ray_intersection(const Sphere &s, const Ray3 &r, float *t1,
                             float *t2) {
  float a = dot(r.d, r.d);
  float b = 2.f * dot(r.d, r.o - s.c);
  float c = dot(r.o - s.c, r.o - s.c) - s.r * s.r;

  float delta = b * b - 4 * a * c;
  if (delta < 0)
    return false;

  float d = sqrt(delta);

  float ta = (-b + d) / (2.0 * a);
  float tb = (-b - d) / (2.0 * a);
  if (ta < 0.f && tb < 0.f)
    return false;
  else if (ta >= 0.f && tb >= 0.f) {
    if (t1)
      *t1 = ta;
    if (t2)
      *t2 = tb;
  } else if (ta >= 0.f) {
    if (t1)
      *t1 = ta;
  } else if (t1)
    *t1 = tb;
  return true;
}

bool bbox_ray_intersection(const BBox2D &box, const Ray2 &ray, float &hit0,
                           float &hit1) {
  float t0 = 0.f, t1 = INFINITY;
  for (int i = 0; i < 2; i++) {
    float invRayDir = 1.f / ray.d[i];
    float tNear = (box.pMin[i] - ray.o[i]) * invRayDir;
    float tFar = (box.pMax[i] - ray.o[i]) * invRayDir;
    if (tNear > tFar)
      std::swap(tNear, tFar);
    t0 = tNear > t0 ? tNear : t0;
    t1 = tFar < t1 ? tFar : t1;
    if (t0 > t1)
      return false;
  }
  hit0 = t0;
  hit1 = t1;
  return true;
}

bool bbox_ray_intersection(const BBox &box, const Ray3 &ray, float &hit0,
                           float &hit1) {
  float t0 = 0.f, t1 = INFINITY;
  for (int i = 0; i < 3; i++) {
    float invRayDir = 1.f / ray.d[i];
    float tNear = (box.pMin[i] - ray.o[i]) * invRayDir;
    float tFar = (box.pMax[i] - ray.o[i]) * invRayDir;
    if (tNear > tFar)
      std::swap(tNear, tFar);
    t0 = tNear > t0 ? tNear : t0;
    t1 = tFar < t1 ? tFar : t1;
    if (t0 > t1)
      return false;
  }
  hit0 = t0;
  hit1 = t1;
  return true;
}

bool bbox_ray_intersection(const BBox &box, const Ray3 &ray, float &hit0) {
  return bbox_ray_intersection(box, ray, hit0);
}

bool triangle_ray_intersection(const Point3 &p1, const Point3 &p2,
                               const Point3 &p3, const Ray3 &ray, float *tHit,
                               float *b1, float *b2) {
  // Compute s1
  vec3 e1 = p2 - p1;
  vec3 e2 = p3 - p1;
  vec3 s1 = cross(ray.d, e2);
  float divisor = dot(s1, e1);
  if (divisor == 0.f)
    return false;
  float invDivisor = 1.f / divisor;
  // Compute first barycentric coordinate
  vec3 d = ray.o - p1;
  float b1_ = dot(d, s1) * invDivisor;
  if (b1_ < 0.f || b1_ > 1.f)
    return false;
  // Compute second barycentric coordinate
  vec3 s2 = cross(d, e1);
  float b2_ = dot(ray.d, s2) * invDivisor;
  if (b2_ < 0. || b1_ + b2_ > 1.)
    return false;
  // Compute t to intersection point
  float t = dot(e2, s2) * invDivisor;
  if (t < 0.)
    return false;
  if (tHit != nullptr)
    *tHit = t;
  if (b1)
    *b1 = b1_;
  if (b2)
    *b2 = b2_;
  return true;
}

Point3 closest_point_plane(const Point3 &p, const Plane &pl) {
  float t =
      (dot(static_cast<vec3>(pl.normal), static_cast<vec3>(p)) - pl.offset) /
      dot(static_cast<vec3>(pl.normal), static_cast<vec3>(pl.normal));
  return p - t * static_cast<vec3>(pl.normal);
}

Point3 closest_point_n_plane(const Point3 &p, const Plane &pl) {
  float t = dot(static_cast<vec3>(pl.normal), static_cast<vec3>(p)) - pl.offset;
  return p - t * static_cast<vec3>(pl.normal);
}

Point3 closest_point_bbox(const Point3 &p, const BBox &b) {
  Point3 cp;
  for (int i = 0; i < 3; i++) {
    float v = p[i];
    if (v < b.pMin[i])
      v = b.pMin[i];
    if (v > b.pMax[i])
      v = b.pMax[i];
    cp[i] = v;
  }
  return cp;
}

float distance_point_plane(const Point3 &p, const Plane &pl) {
  return (dot(static_cast<vec3>(pl.normal), static_cast<vec3>(p)) - pl.offset) /
         dot(static_cast<vec3>(pl.normal), static_cast<vec3>(pl.normal));
}

float distance_point_n_plane(const Point3 &p, const Plane &pl) {
  return dot(static_cast<vec3>(pl.normal), static_cast<vec3>(p)) - pl.offset;
}

template <typename T>
float distance2_point_segment(const T &p, const Segment<T> &s) {
  float e = dot(p - s.a, s.b - s.a);
  if (e <= 0.f)
    return (p - s.a).length2();
  float f = s.length2();
  if (e <= f)
    return (p - s.b).length2();
  return s.length2() - e * e / f;
}

float distance2_point_bbox(const Point3 &p, const BBox &b) {
  float sdist = 0.f;
  for (int i = 0; i < 3; i++) {
    float v = p[i];
    if (v < b.pMin[i])
      sdist += SQR(b.pMin[i] - v);
    if (v > b.pMax[i])
      sdist += SQR(v - b.pMax[i]);
  }
  return sdist;
}

} // ponos namespace
