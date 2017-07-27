#ifndef PONOS_GEOMETRY_INTERSECTIONS_H
#define PONOS_GEOMETRY_INTERSECTIONS_H

#include "geometry/bbox.h"
#include "geometry/interval.h"
#include "geometry/line.h"
#include "geometry/plane.h"
#include "geometry/polygon.h"
#include "geometry/ray.h"
#include "geometry/segment.h"
#include "geometry/sphere.h"
#include "geometry/transform.h"
#include "geometry/utils.h"
#include "geometry/vector.h"

namespace ponos {

/** \brief  intersection test
 * \param a **[in]**
 * \param b **[in]**
 * /return
 */
bool bbox_bbox_intersection(const BBox2D &a, const BBox2D &b);
/** \brief  intersection test
* \param a **[in]**
* \param b **[in]**
* /return
*/
template <typename T>
bool interval_interval_intersection(const Interval<T> &a,
                                    const Interval<T> &b) {
  return (a.low <= b.low && a.high >= b.low) ||
         (b.low <= a.low && b.high >= a.low);
}

/** \brief  intersection test
 * \param r **[in]** ray
 * \param s **[in]** segment
 * \param t **[out]** intersection point (ray coordinate)
 * /return **true** if intersection exists
 */
bool ray_segment_intersection(const Ray2 &r, const Segment2 &s,
                              float *t = nullptr);

/** \brief  intersection test
 * \param r **[in]** ray
 * \param s **[in]** segment
 * \param t **[out]** intersection point (ray coordinate)
 * /return **true** if intersection exists
 */
bool ray_segment_intersection(const Ray3 &r, const Segment3 &s,
                              float *t = nullptr);
/** \brief  intersection test
 * \param pl **[in]** plane
 * \param l **[in]** line
 * \param p **[out]** intersection point
 *
 * <Plane> / <Line> intersection test
 *
 * /return **true** if intersection exists
 */
bool plane_line_intersection(const Plane pl, const Line l, Point3 &p);
/** \brief  intersection test
 * \param s **[in]** sphere
 * \param l **[in]** line
 * \param p **[out]** intersection point
 * \param p1 **[out]** first intersection
 * \param p2 **[out]** second intersection
 *
 * <Sphere> / <Line> intersection test
 *
 * **p1** = **p2** if line is tangent to sphere
 *
 * /return **true** if intersection exists
 */
bool sphere_line_intersection(const Sphere s, const Line l, Point3 &p1,
                              Point3 &p2);
/** \brief  intersection test
 * \param s **[in]** sphere
 * \param r **[in]** ray
 * \param t1 **[out | optional]** closest intersection (parametric coordinate)
 * \param t2 **[out | optional]** second intersection (parametric coordinate)
 *
 * /return **true** if intersection exists
 */
bool sphere_ray_intersection(const Sphere &s, const Ray3 &r,
                             float *t1 = nullptr, float *t2 = nullptr);
/** \brief  intersection test
 * \param box **[in]**
 * \param ray **[in]**
 * \param hit1 **[out]** first intersection
 * \param hit2 **[out]** second intersection
 * \param normal **[out | optional]** collision normal
 *
 * <BBox2D> / <Ray> intersection test.
 *
 * **hit1** and **hit2** are in the ray's parametric coordinate.
 *
 * **hit1** = **hit2** if a single point is found.
 *
 * /return **true** if intersectiton exists
 */
bool bbox_ray_intersection(const BBox2D &box, const Ray2 &ray, float &hit1,
                           float &hit2, float *normal = nullptr);

/** \brief  intersection test
 * \param box **[in]**
 * \param ray **[in]**
 * \param hit1 **[out]** first intersection
 * \param hit2 **[out]** second intersection
 *
 * <BBox> / <Ray3> intersection test.
 *
 * **hit1** and **hit2** are in the ray's parametric coordinate.
 *
 * **hit1** = **hit2** if a single point is found.
 *
 * /return **true** if intersectiton exists
 */
bool bbox_ray_intersection(const BBox &box, const Ray3 &ray, float &hit1,
                           float &hit2);
/** \brief  intersection test
 * \param box **[in]**
 * \param ray **[in]**
 * \param hit1 **[out]** closest intersection
 *
 * <BBox> / <Ray3> intersection test. Computes the closest intersection from the
 *ray's origin.
 *
 * **hit1** in in the ray's parametric coordinate.
 *
 * /return **true** if intersection exists
 */
void triangle_barycentric_coordinates(const Point2 &p, const Point2 &a,
                                      const Point2 &b, const Point2 &c,
                                      float &u, float &v, float &w);
bool bbox_ray_intersection(const BBox &box, const Ray3 &ray, float &hit1);
bool triangle_point_intersection(const Point2 &p, const Point2 *vertices);
/** \brief  intersection test
 * \param p1 **[in]** first triangle's vertex
 * \param p2 **[in]** second triangle's vertex
 * \param p3 **[in]** third triangle's vertex
 * \param ray **[in]**
 * \param tHit **[out]** intersection (parametric coordinate)
 * \param b1 **[out]** barycentric coordinate
 * \param b2 **[out]** barycentric coordinate
 *
 * <BBox> / <Ray3> intersection test
 *
 * **b1** and **b2**, if not null, receive the barycentric coordinates of the
 *intersection point.
 *
 * /return **true** if intersection exists
 */
bool triangle_ray_intersection(const Point3 &p1, const Point3 &p2,
                               const Point3 &p3, const Ray3 &ray,
                               float *tHit = nullptr, float *b1 = nullptr,
                               float *b2 = nullptr);
/** \brief  closest point
 * \param p **[in]** point
 * \param pl **[in]** plane
 *
 * /return closest point on **pl** from **p**
 */
Point3 closest_point_plane(const Point3 &p, const Plane &pl);
/** \brief
 * \param s **[in]** segment
 * \param p **[in]** point
 * \param t **[in]** receives
 *
 * If the projected **p** on **s** lies outside **s**, **t** is clamped to
 *[0,1].
 *
 * /return closest point to **p** that lies on **s**
 */
template <typename T>
T closest_point_segment(const Segment<T> &s, const T &p, float *t = nullptr) {
  float t_ = dot(p - s.a, s.b - s.a) / (s.b - s.a).length2();
  if (t_ < 0.f)
    t_ = 0.f;
  if (t_ > 1.f)
    t_ = 1.f;
  if (t != nullptr)
    *t = t_;
  return s.a + t_ * (s.b - s.a);
}
/** \brief  closest point
 * \param p **[in]**
 * \param pl **[in]**
 *
 * Assumes **pl**'s normal is normalized.
 *
 * /return closest point on **pl** from **p**
 */
Point3 closest_point_n_plane(const Point3 &p, const Plane &pl);
/** \brief  closest point
 * \param p **[in]** point
 * \param b **[in]** bbox
 *
 * /return closest point on **b** from **p**
 */
Point3 closest_point_bbox(const Point3 &p, const BBox &b);
/** \brief  distance
 * \param p **[in]** point
 * \param pl **[in]** plane
 *
 * /return signed distance of **p** to **pl**
 */
float distance_point_plane(const Point3 &p, const Plane &pl);
/** \brief  distance
 * \param p **[in]** point
 * \param pl **[in]** plane
 *
 * Assumes **pl**'s normal is normalized.
 *
 * /return signed distance of **p** to **pl**
 */
float distance_point_n_plane(const Point3 &p, const Plane &pl);
/** \brief  distance
 * \param p **[in]** point
 * \param s **[in]** segment
 *
 * /return the squared distance between **p** and **s**
 */
template <typename T>
float distance2_point_segment(const T &p, const Segment<T> &s);
/** \brief  distance
 * \param p **[in]** point
 * \param b **[in]** bbox
 *
 * /return squared distance between **p** and **b**
 */
float distance2_point_bbox(const Point3 &p, const BBox &b);

inline BBox2D compute_bbox(const Shape &po, const Transform2D *t = nullptr) {
  BBox2D b;
  if (po.type == ShapeType::POLYGON)
    b = compute_bbox(static_cast<const Polygon &>(po), t);
  else if (po.type == ShapeType::SPHERE)
    b = compute_bbox(static_cast<const Circle &>(po), t);
  return b;
}
} // ponos namespace

#endif
