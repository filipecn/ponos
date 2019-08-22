#include <ponos/geometry/matrix.h>
#include <ponos/geometry/numeric.h>
#include <ponos/geometry/queries.h>

#include <utility>

namespace ponos {

bool bbox_bbox_intersection(const bbox2 &a, const bbox2 &b) {
  for (int i = 0; i < 2; i++)
    if (!((a.lower[i] <= b.lower[i] && a.upper[i] >= b.lower[i]) ||
          (b.lower[i] <= a.lower[i] && b.upper[i] >= a.lower[i])))
      return false;
  return true;
}

bool bbox_bbox_intersection(const bbox3 &a, const bbox3 &b) {
  for (int i = 0; i < 3; i++)
    if (!((a.lower[i] <= b.lower[i] && a.upper[i] >= b.lower[i]) ||
          (b.lower[i] <= a.lower[i] && b.upper[i] >= a.lower[i])))
      return false;
  return true;
}

bool ray_segment_intersection(const Ray2 &R, const Segment2 &s, real_t *t) {
  vec2 dd = s.b - s.a;
  Ray3 r(point3(R.o.x, R.o.y, 0.f), vec3(R.d.x, R.d.y, 0.f));
  Ray3 r2(point3(s.a.x, s.a.y, 0.f), vec3(dd.x, dd.y, 0.f));
  real_t d = cross(r.d, r2.d).length2();
  if (d == 0.f)
    // TODO handle this case
    return false;
  real_t t1 = mat3(r2.o - r.o, r2.d, cross(r.d, r2.d)).determinant() / d;
  real_t t2 = mat3(r2.o - r.o, r.d, cross(r.d, r2.d)).determinant() / d;
  // TODO add user threshold
  real_t dist = (r(t1) - r2(t2)).length();
  if (t1 >= 0.f && t2 >= 0.f && t2 <= s.length() && dist < 0.1f) {
    if (t)
      *t = t1;
    return true;
  }
  return false;
}

bool ray_segment_intersection(const Ray3 &r, const Segment3 &s, real_t *t) {
  Ray3 r2(s.a, s.b - s.a);
  real_t d = cross(r.d, r2.d).length2();
  if (d == 0.f)
    // TODO handle this case
    return false;
  real_t t1 = mat3(r2.o - r.o, r2.d, cross(r.d, r2.d)).determinant() / d;
  real_t t2 = mat3(r2.o - r.o, r.d, cross(r.d, r2.d)).determinant() / d;
  // TODO add user threshold
  real_t dist = (r(t1) - r2(t2)).length();
  if (t1 >= 0.f && t2 >= 0.f && t2 <= s.length() && dist < 0.1f) {
    if (t)
      *t = t1;
    return true;
  }
  return false;
}

bool plane_line_intersection(const Plane pl, const Line l, point3 &p) {
  vec3 nvector = vec3(pl.normal.x, pl.normal.y, pl.normal.z);
  real_t k = dot(nvector, l.direction());
  if (Check::is_zero(k))
    return false;
  real_t r = (pl.offset - dot(nvector, vec3(l.a.x, l.a.y, l.a.z))) / k;
  p = l(r);
  return true;
}

bool sphere_line_intersection(const Sphere s, const Line l, point3 &p1,
                              point3 &p2) {
  real_t a = dot(l.direction(), l.direction());
  real_t b = 2.f * dot(l.direction(), l.a - s.c);
  real_t c = dot(l.a - s.c, l.a - s.c) - s.r * s.r;

  real_t delta = b * b - 4 * a * c;
  if (delta < 0)
    return false;

  real_t d = sqrt(delta);

  p1 = l((-b + d) / (2.0 * a));
  p2 = l((-b - d) / (2.0 * a));
  return true;
}

bool sphere_ray_intersection(const Sphere &s, const Ray3 &r, real_t *t1,
                             real_t *t2) {
  real_t a = dot(r.d, r.d);
  real_t b = 2.f * dot(r.d, r.o - s.c);
  real_t c = dot(r.o - s.c, r.o - s.c) - s.r * s.r;

  real_t delta = b * b - 4 * a * c;
  if (delta < 0)
    return false;

  real_t d = sqrt(delta);

  real_t ta = (-b + d) / (2.0 * a);
  real_t tb = (-b - d) / (2.0 * a);
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

bool bbox_ray_intersection(const bbox2 &box, const Ray2 &ray, real_t &hit0,
                           real_t &hit1, real_t *normal) {
  int sign[2];
  vec2 invdir = 1.f / ray.d;
  sign[0] = (invdir.x < 0);
  sign[1] = (invdir.y < 0);
  real_t tmin = (box[sign[0]].x - ray.o.x) * invdir.x;
  real_t tmax = (box[1 - sign[0]].x - ray.o.x) * invdir.x;
  real_t tymin = (box[sign[1]].y - ray.o.y) * invdir.y;
  real_t tymax = (box[1 - sign[1]].y - ray.o.y) * invdir.y;

  if ((tmin > tymax) || (tymin > tmax))
    return false;
  if (tymin > tmin)
    tmin = tymin;
  if (tymax < tmax)
    tmax = tymax;
  if (normal) {
    if (ray.o.x < box[0].x && ray.o.y >= box[0].y && ray.o.y <= box[1].y) {
      normal[0] = -1;
      normal[1] = 0;
    } else if (ray.o.x > box[1].x && ray.o.y >= box[0].y &&
               ray.o.y <= box[1].y) {
      normal[0] = 1;
      normal[1] = 0;
    } else if (ray.o.y < box[0].y && ray.o.x >= box[0].x &&
               ray.o.x <= box[1].x) {
      normal[0] = 0;
      normal[1] = -1;
    } else if (ray.o.y > box[0].y && ray.o.x >= box[0].x &&
               ray.o.x <= box[1].x) {
      normal[0] = 0;
      normal[1] = 1;
    } else if (ray.o.x < box[0].x && ray.o.y < box[0].y) {
      normal[0] = -1;
      normal[1] = -1;
    } else if (ray.o.x < box[0].x && ray.o.y > box[1].y) {
      normal[0] = -1;
      normal[1] = 1;
    } else if (ray.o.x > box[0].x && ray.o.y > box[1].y) {
      normal[0] = 1;
      normal[1] = 1;
    } else {
      normal[0] = 1;
      normal[1] = -1;
    }
  }
  hit0 = tmin;
  hit1 = tmax;
  return true;
  /*for (int i = 0; i < 2; i++) {
    real_t invRayDir = 1.f / ray.d[i];
    real_t tNear = (box.lower[i] - ray.o[i]) * invRayDir;
    real_t tFar = (box.upper[i] - ray.o[i]) * invRayDir;
    if (tNear > tFar)
      std::swap(tNear, tFar);
    t0 = tNear > t0 ? tNear : t0;
    t1 = tFar < t1 ? tFar : t1;
    if (t0 > t1)
      return false;
  }
  hit0 = t0;
  hit1 = t1;
  return true;*/
}

bool bbox_ray_intersection(const bbox3 &box, const Ray3 &ray, real_t &hit0,
                           real_t &hit1) {
  real_t t0 = 0.f, t1 = INFINITY;
  for (int i = 0; i < 3; i++) {
    real_t invRayDir = 1.f / ray.d[i];
    real_t tNear = (box.lower[i] - ray.o[i]) * invRayDir;
    real_t tFar = (box.upper[i] - ray.o[i]) * invRayDir;
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

bool bbox_ray_intersection(const bbox3 &box, const Ray3 &ray, real_t &hit0) {
  real_t hit1 = 0;
  return bbox_ray_intersection(box, ray, hit0, hit1);
}

void triangle_barycentric_coordinates(const point2 &p, const point2 &a,
                                      const point2 &b, const point2 &c,
                                      real_t &u, real_t &v, real_t &w) {
  vec2 v0 = b - a, v1 = c - a, v2 = p - a;
  real_t d00 = dot(v0, v0);
  real_t d01 = dot(v0, v1);
  real_t d11 = dot(v1, v1);
  real_t d20 = dot(v2, v0);
  real_t d21 = dot(v2, v1);
  real_t d = d00 * d11 - d01 * d01;
  v = (d11 * d20 - d01 * d21) / d;
  w = (d00 * d21 - d01 * d20) / d;
  u = 1.0f - v - w;
}

bool triangle_point_intersection(const point2 &p, const point2 *vertices) {
  real_t u = -1, v = -1, w = -1;
  triangle_barycentric_coordinates(p, vertices[0], vertices[1], vertices[2], u,
                                   v, w);
  return Check::is_between_closed(u, 0.f, 1.f) &&
         Check::is_between_closed(v, 0.f, 1.f) &&
         Check::is_between_closed(w, 0.f, 1.f);
}

bool triangle_ray_intersection(const point3 &p1, const point3 &p2,
                               const point3 &p3, const Ray3 &ray, real_t *tHit,
                               real_t *b1, real_t *b2) {
  // Compute s1
  vec3 e1 = p2 - p1;
  vec3 e2 = p3 - p1;
  vec3 s1 = cross(ray.d, e2);
  real_t divisor = dot(s1, e1);
  if (divisor == 0.f)
    return false;
  real_t invDivisor = 1.f / divisor;
  // Compute first barycentric coordinate
  vec3 d = ray.o - p1;
  real_t b1_ = dot(d, s1) * invDivisor;
  if (b1_ < 0.f || b1_ > 1.f)
    return false;
  // Compute second barycentric coordinate
  vec3 s2 = cross(d, e1);
  real_t b2_ = dot(ray.d, s2) * invDivisor;
  if (b2_ < 0. || b1_ + b2_ > 1.)
    return false;
  // Compute t to intersection point
  real_t t = dot(e2, s2) * invDivisor;
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

point3 closest_point_triangle(const point3 &p, const point3 &p1,
                              const point3 &p2, const point3 &p3) {
  auto ab = p2 - p1;
  auto ac = p3 - p1;
  auto ap = p - p1;
  real_t d1 = dot(ab, ap);
  real_t d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f)
    return p1;
  auto bp = p - p2;
  real_t d3 = dot(ab, bp);
  real_t d4 = dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3)
    return p2;
  real_t vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    real_t v = d1 / (d1 - d3);
    return p1 + v * ab;
  }
  auto cp = p - p3;
  real_t d5 = dot(ab, cp);
  real_t d6 = dot(ac, cp);
  if (d6 >= 0.f && d5 <= d6)
    return p3;
  real_t vb = d5 * d2 - d1 * d6;
  if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
    real_t w = d2 / (d2 - d6);
    return p1 + w * ac;
  }
  real_t va = d3 * d6 - d5 * d4;
  if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
    real_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return p2 + w * (p3 - p2);
  }
  real_t den = 1.f / (va + vb + vc);
  real_t v = vb * den;
  real_t w = vc * den;
  return p1 + ab * v + ac * w;
}

point3 closest_point_plane(const point3 &p, const Plane &pl) {
  real_t t =
      (dot(static_cast<vec3>(pl.normal), static_cast<vec3>(p)) - pl.offset) /
      dot(static_cast<vec3>(pl.normal), static_cast<vec3>(pl.normal));
  return p - t * static_cast<vec3>(pl.normal);
}

point3 closest_point_n_plane(const point3 &p, const Plane &pl) {
  real_t t =
      dot(static_cast<vec3>(pl.normal), static_cast<vec3>(p)) - pl.offset;
  return p - t * static_cast<vec3>(pl.normal);
}

point3 closest_point_bbox(const point3 &p, const bbox3 &b) {
  point3 cp;
  for (int i = 0; i < 3; i++) {
    real_t v = p[i];
    if (v < b.lower[i])
      v = b.lower[i];
    if (v > b.upper[i])
      v = b.upper[i];
    cp[i] = v;
  }
  return cp;
}

real_t distance_point_line(const point3 &p, const Line &l) {
  return distance(p, l.closestPoint(p));
}

real_t distance_point_plane(const point3 &p, const Plane &pl) {
  return (dot(static_cast<vec3>(pl.normal), static_cast<vec3>(p)) - pl.offset) /
         dot(static_cast<vec3>(pl.normal), static_cast<vec3>(pl.normal));
}

real_t distance_point_n_plane(const point3 &p, const Plane &pl) {
  return dot(static_cast<vec3>(pl.normal), static_cast<vec3>(p)) - pl.offset;
}

template <typename T>
real_t distance2_point_segment(const T &p, const Segment<T> &s) {
  real_t e = dot(p - s.a, s.b - s.a);
  if (e <= 0.f)
    return (p - s.a).length2();
  real_t f = s.length2();
  if (e <= f)
    return (p - s.b).length2();
  return s.length2() - e * e / f;
}

real_t distance2_point_bbox(const point3 &p, const bbox3 &b) {
  real_t sdist = 0.f;
  for (int i = 0; i < 3; i++) {
    real_t v = p[i];
    if (v < b.lower[i])
      sdist += SQR(b.lower[i] - v);
    if (v > b.upper[i])
      sdist += SQR(v - b.upper[i]);
  }
  return sdist;
}

} // namespace ponos
