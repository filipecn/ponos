#ifndef PONOS_GEOMETRY_INTERSECTIONS_H
#define PONOS_GEOMETRY_INTERSECTIONS_H

#include "geometry/bbox.h"
#include "geometry/interval.h"
#include "geometry/line.h"
#include "geometry/plane.h"
#include "geometry/ray.h"
#include "geometry/segment.h"
#include "geometry/sphere.h"
#include "geometry/utils.h"
#include "geometry/vector.h"
#include "geometry/transform.h"
#include "geometry/polygon.h"

namespace ponos {

	/* intersection test
	 * @a **[in]**
	 * @b **[in]**
	 * @return
	 */
	bool bbox_bbox_intersection(const BBox2D& a, const BBox2D& b);
		/* intersection test
	 * @a **[in]**
	 * @b **[in]**
	 * @return
	 */
	template<typename T>
		bool interval_interval_intersection(const Interval<T>& a, const Interval<T>& b) {
			return (a.low <= b.low && a.high >= b.low) || (b.low <= a.low && b.high >= a.low);
		}

	/* intersection test
	 * @r **[in]** ray
	 * @s **[in]** segment
	 * @t **[out]** intersection point (ray coordinate)
	 * @return **true** if intersection exists
	 */
	bool ray_segment_intersection(const Ray2& R, const Segment2& s, float *t = nullptr);

	/* intersection test
	 * @r **[in]** ray
	 * @s **[in]** segment
	 * @t **[out]** intersection point (ray coordinate)
	 * @return **true** if intersection exists
	 */
	bool ray_segment_intersection(const Ray3& r, const Segment3& s, float *t = nullptr);
	/* intersection test
	 * @pl **[in]** plane
	 * @l **[in]** line
	 * @p **[out]** intersection point
	 *
	 * <Plane> / <Line> intersection test
	 *
	 * @return **true** if intersection exists
	 */
	bool plane_line_intersection(const Plane pl, const Line l, Point3& p);
	/* intersection test
	 * @s **[in]** sphere
	 * @l **[in]** line
	 * @p **[out]** intersection point
	 * @p1 **[out]** first intersection
	 * @p2 **[out]** second intersection
	 *
	 * <Sphere> / <Line> intersection test
	 *
	 * **p1** = **p2** if line is tangent to sphere
	 *
	 * @return **true** if intersection exists
	 */
	bool sphere_line_intersection(const Sphere s, const Line l, Point3& p1, Point3& p2);
	/* intersection test
	 * @s **[in]** sphere
	 * @r **[in]** ray
	 * @t1 **[out | optional]** closest intersection (parametric coordinate)
	 * @t2 **[out | optional]** second intersection (parametric coordinate)
	 *
	 * @return **true** if intersection exists
	 */
	bool sphere_ray_intersection(const Sphere &s, const Ray3 &r, float *t1 = nullptr, float *t2 = nullptr);
	/* intersection test
	 * @box **[in]**
	 * @ray **[in]**
	 * @hit1 **[out]** first intersection
	 * @hit2 **[out]** second intersection
	 *
	 * <BBox2D> / <Ray> intersection test.
	 *
	 * **hit1** and **hit2** are in the ray's parametric coordinate.
	 *
	 * **hit1** = **hit2** if a single point is found.
	 *
	 * @return **true** if intersectiton exists
	 */
	bool bbox_ray_intersection(const BBox2D& box, const Ray2& ray, float& hit1, float& hit2);

	/* intersection test
	 * @box **[in]**
	 * @ray **[in]**
	 * @hit1 **[out]** first intersection
	 * @hit2 **[out]** second intersection
	 *
	 * <BBox> / <Ray3> intersection test.
	 *
	 * **hit1** and **hit2** are in the ray's parametric coordinate.
	 *
	 * **hit1** = **hit2** if a single point is found.
	 *
	 * @return **true** if intersectiton exists
	 */
	bool bbox_ray_intersection(const BBox& box, const Ray3& ray, float& hit1, float& hit2);
	/* intersection test
	 * @box **[in]**
	 * @ray **[in]**
	 * @hit1 **[out]** closest intersection
	 *
	 * <BBox> / <Ray3> intersection test. Computes the closest intersection from the ray's origin.
	 *
	 * **hit1** in in the ray's parametric coordinate.
	 *
	 * @return **true** if intersection exists
	 */
	bool bbox_ray_intersection(const BBox& box, const Ray3& ray, float& hit1);
	/* intersection test
	 * @p1 **[in]** first triangle's vertex
	 * @p2 **[in]** second triangle's vertex
	 * @p3 **[in]** third triangle's vertex
	 * @ray **[in]**
	 * @tHit **[out]** intersection (parametric coordinate)
	 * @b1 **[out]** barycentric coordinate
	 * @b2 **[out]** barycentric coordinate
	 *
	 * <BBox> / <Ray3> intersection test
	 *
	 * **b1** and **b2**, if not null, receive the barycentric coordinates of the intersection point.
	 *
	 * @return **true** if intersection exists
	 */
	bool triangle_ray_intersection(const Point3& p1, const Point3& p2, const Point3& p3, const Ray3& ray, float *tHit = nullptr, float *b1 = nullptr, float *b2 = nullptr);
	/* closest point
	 * @p **[in]** point
	 * @pl **[in]** plane
	 *
	 * @return closest point on **pl** from **p**
	 */
	Point3 closest_point_plane(const Point3& p, const Plane& pl);
	/*
	 * @s **[in]** segment
	 * @p **[in]** point
	 * @t **[in]** receives
	 *
	 * If the projected **p** on **s** lies outside **s**, **t** is clamped to [0,1].
	 *
	 * @return closest point to **p** that lies on **s**
	 */
	template<typename T>
		T closest_point_segment(const Segment<T>& s, const T& p, float* t = nullptr) {
			float t_ = dot(p - s.a, s.b - s.a) / (s.b - s.a).length2();
			if(t_ < 0.f) t_ = 0.f;
			if(t_ > 1.f) t_ = 1.f;
			if(t != nullptr)
				*t = t_;
			return s.a + t_ * (s.b - s.a);
		}
	/* closest point
	 * @p **[in]**
	 * @pl **[in]**
	 *
	 * Assumes **pl**'s normal is normalized.
	 *
	 * @return closest point on **pl** from **p**
	 */
	Point3 closest_point_n_plane(const Point3& p, const Plane& pl);
	/* closest point
	 * @p **[in]** point
	 * @b **[in]** bbox
	 *
	 * @return closest point on **b** from **p**
	 */
	Point3 closest_point_bbox(const Point3& p, const BBox& b);
	/* distance
	 * @p **[in]** point
	 * @pl **[in]** plane
	 *
	 * @return signed distance of **p** to **pl**
	 */
	float distance_point_plane(const Point3& p, const Plane& pl);
	/* distance
	 * @p **[in]** point
	 * @pl **[in]** plane
	 *
	 * Assumes **pl**'s normal is normalized.
	 *
	 * @return signed distance of **p** to **pl**
	 */
	float distance_point_n_plane(const Point3& p, const Plane& pl);
	/* distance
	 * @p **[in]** point
	 * @s **[in]** segment
	 *
	 * @return the squared distance between **p** and **s**
	 */
	template<typename T>
		float distance2_point_segment(const T& p, const Segment<T>& s);
	/* distance
	 * @p **[in]** point
	 * @b **[in]** bbox
	 *
	 * @return squared distance between **p** and **b**
	 */
	float distance2_point_bbox(const Point3& p, const BBox& b);

  inline BBox2D compute_bbox(const Shape& po, const Transform2D* t = nullptr) {
    BBox2D b;
    if(po.type == ShapeType::POLYGON)
      b = compute_bbox(static_cast<const Polygon&>(po), t);
    else if(po.type == ShapeType::SPHERE)
      b = compute_bbox(static_cast<const Circle&>(po), t);
    return b;
  }
} // ponos namespace

#endif
