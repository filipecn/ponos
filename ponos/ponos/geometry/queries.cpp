#include <ponos/geometry/matrix.h>
#include <ponos/numeric/numeric.h>
#include <ponos/geometry/queries.h>

#include <utility>

namespace ponos {

point3 GeometricQueries::closestPoint(const bbox3 &box, const point3 &p) {

  return ponos::point3();
}

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

bool plane_line_intersection(const Plane &pl, const Line &l, point3 &p) {
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

template<typename T>
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

std::optional<real_t> GeometricPredicates::intersect(const bbox3 &bounds,
                                                     const ray3 &ray,
                                                     const vec3 &inv_dir,
                                                     const i32 *dir_is_neg,
                                                     real_t max_t) {
  // check for ray intersection against x and y slabs
  real_t t_min = (bounds[dir_is_neg[0]].x - ray.o.x) * inv_dir.x;
  real_t t_max = (bounds[1 - dir_is_neg[0]].x - ray.o.x) * inv_dir.x;
  real_t ty_min = (bounds[dir_is_neg[1]].y - ray.o.y) * inv_dir.y;
  real_t ty_max = (bounds[1 - dir_is_neg[1]].y - ray.o.y) * inv_dir.y;
  // update t_max and tyMax to ensure robust bounds intersection
  if (t_min > ty_max || ty_min > t_max)
    return std::nullopt;
  if (ty_min > t_min)
    t_min = ty_min;
  if (ty_max < t_max)
    t_max = ty_max;
  // check for ray intersection against z slab
  real_t tz_min = (bounds[dir_is_neg[2]].z - ray.o.z) * inv_dir.z;
  real_t tz_max = (bounds[1 - dir_is_neg[2]].z - ray.o.z) * inv_dir.z;
  if (t_min > tz_max || tz_min > t_max)
    return std::nullopt;
  if (tz_min > t_min)
    t_min = tz_min;
  if (tz_max < t_max)
    t_max = tz_max;
  if ((t_min < max_t) && (t_max > 0))
    return std::optional<real_t>{t_min};
  return std::nullopt;
}

std::optional<real_t> GeometricPredicates::intersect(const ponos::bbox3 &bounds,
                                                     const ray3 &ray, real_t *second_hit) {

  real_t t0 = 0.f, t1 = Constants::real_infinity;
  for (int i = 0; i < 3; i++) {
    real_t inv_ray_dir = 1.f / ray.d[i];
    real_t t_near = (bounds.lower[i] - ray.o[i]) * inv_ray_dir;
    real_t t_far = (bounds.upper[i] - ray.o[i]) * inv_ray_dir;
    if (t_near > t_far)
      std::swap(t_near, t_far);
    t0 = t_near > t0 ? t_near : t0;
    t1 = t_far < t1 ? t_far : t1;
    if (t0 > t1)
      return std::nullopt;
  }
  std::optional<real_t> hit(t0);
  if (second_hit)
    *second_hit = t1;
  return hit;
}

std::optional<real_t> GeometricPredicates::intersect(const point3 &p1,
                                                     const point3 &p2,
                                                     const point3 &p3,
                                                     const Ray3 &ray,
                                                     real_t *b0,
                                                     real_t *b1) {
  real_t max_t = Constants::real_infinity;
  // transform triangle vertices to ray coordinate space
  //    translate vertices based on ray origin
  auto p0t = p1 - vec3(ray.o);
  auto p1t = p2 - vec3(ray.o);
  auto p2t = p3 - vec3(ray.o);
  //    permute components of triangle vertices and ray direction
  int kz = abs(ray.d).maxDimension();
  int kx = (kz + 1) % 3;
  int ky = (kx + 1) % 3;
  vec3 d = {ray.d[kx], ray.d[ky], ray.d[kz]};
  p0t = {p0t[kx], p0t[ky], p0t[kz]};
  p1t = {p1t[kx], p1t[ky], p1t[kz]};
  p2t = {p2t[kx], p2t[ky], p2t[kz]};
  //    align ray direction with the z+ axis
  real_t sx = -d.x / d.z;
  real_t sy = -d.y / d.z;
  real_t sz = 1.f / d.z;
  p0t.x += sx * p0t.z;
  p0t.y += sy * p0t.z;
  p1t.x += sx * p1t.z;
  p1t.y += sy * p1t.z;
  p2t.x += sx * p2t.z;
  p2t.y += sy * p2t.z;
  // compute edge function coefficients e0, e1, and e2
  real_t e0 = p1t.x * p2t.y - p1t.y * p2t.x;
  real_t e1 = p2t.x * p0t.y - p2t.y * p0t.x;
  real_t e2 = p0t.x * p1t.y - p0t.y * p1t.x;
  // fall back to double precision test at triangle edges
  if (e0 == 0 && e1 == 0 && e2 == 0) {
    e0 = static_cast<f64>(p1t.x) * p2t.y - static_cast<f64>(p1t.y) * p2t.x;
    e1 = static_cast<f64>(p2t.x) * p0t.y - static_cast<f64>(p2t.y) * p0t.x;
    e2 = static_cast<f64>(p0t.x) * p1t.y - static_cast<f64>(p0t.y) * p1t.x;
  }
  // perform triangle edge and determinant tests
  if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
    return std::nullopt;
  real_t det = e0 + e1 + e2;
  if (det == 0)
    return std::nullopt;
  // compute scaled hit distance to triangle and test against ray t range
  p0t.z *= sz;
  p1t.z *= sz;
  p2t.z *= sz;
  real_t t_scaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
  if ((det < 0 && (t_scaled >= 0 || t_scaled < max_t * det)) ||
      (det < 0 && (t_scaled <= 0 || t_scaled > max_t * det)))
    return std::nullopt;
  // compute barycentric coordinates and t value for triangle intersection
  real_t inv_det = 1 / det;
  if (b0)
    *b0 = e0 * inv_det;
  if (b1)
    *b1 = e1 * inv_det;
  //  real_t b2 = e2 * inv_det;
  real_t t = t_scaled * inv_det;
  // ensure that computed triangle t is conservatively greater than zero
  //    compute dz term for triangle t error bounds
  real_t max_z_t = abs(vec3(p0t.z, p1t.z, p2t.z)).maxDimension();
  real_t delta_z = Constants::gamma(3) * max_z_t;
  //    compute dx and dy terms for triangle t error bounds
  real_t max_x_t = abs(vec3(p0t.x, p1t.x, p2t.x)).maxDimension();
  real_t max_y_t = abs(vec3(p0t.y, p1t.y, p2t.y)).maxDimension();
  real_t delta_x = Constants::gamma(5) * (max_x_t + max_z_t);
  real_t delta_y = Constants::gamma(5) * (max_y_t + max_z_t);
  //    compute de term for triangle error bounds
  real_t delta_e = 2 * (Constants::gamma(2) * max_x_t * max_y_t + delta_y * max_x_t + delta_x * max_y_t);
  //    compute dt term for triangle t error bounds and check t
  real_t max_e = abs(vec3(e0, e1, e2)).maxDimension();
  real_t
      delta_t = 3 * (Constants::gamma(3) * max_e * max_z_t + delta_e * max_z_t + delta_z * max_e) * std::abs(inv_det);
  if (t <= delta_t)
    return std::nullopt;
  return std::optional<real_t>(t);
}

} // namespace ponos
