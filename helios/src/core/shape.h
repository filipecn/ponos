#ifndef HELIOS_CORE_SHAPE_H
#define HELIOS_CORE_SHAPE_H

#include "core/differential_geometry.h"
#include "geometry/h_ray.h"
#include <ponos.h>

#include <memory>
#include <vector>

namespace helios {

	/* Geometric shape Interface.
	 * Provides all the geometric information about the <helios::Primitive> such as its surface area and bounding box.
	 * All shapes receive a unique id.
	 */
	class Shape {
		public:
		/* Constructor.
		 * @o2w object to world transformation
		 * @w2o world to object transformation
		 * @ro reverse ortientation: indicates if the surface normal directions should be reversed from default (default = normals pointing outside).
		 */
		Shape(const ponos::Transform *o2w, const ponos::Transform *w2o, bool ro);

		/* Shape bounding box.
		 *
		 * @return bounding box of the shape (in object space)
		 */
		virtual ponos::BBox objectBound() const = 0;
		/* Shape bounding box.
		 *
		 * @return bounding box of the shapet (in world space)
		 */
		ponos::BBox worldBound() const;
		/*
		 */
		virtual void getShadingGeometry(const ponos::Transform &o2w,
				const DifferentialGeometry &dg,
				DifferentialGeometry *dgShading) const {
			*dgShading = dg;
		}

		/* if is intersectable.
		 *
		 * Indicates if the shape can compute <helios::HRay> intersections.
		 */
		bool canIntersect() const;
		/* refine shape.
		 *
		 * Splits shape into a group of new shapes.
		 */
		void refine(std::vector<std::shared_ptr<Shape> > &refined) const;
		/* compute intersection.
		 * @ray ray to be intersected (in world space).
		 * @tHit if an intersection is found, **tHit receives the parametric distance along the ray to the intersection point.
		 * @rayEpsilon if an intersection is found, **rayEpsilon is initialized with  the maximum numeric error in the intersection calculation (it helps the ray tracing algorithms to avoid incorrect self-intersections)
		 * @gd the geometric propertes of the surface in the point of the intersection
		 *
		 * The rays passed into intersection routines are in world space!
		 *
		 * @return true if an intersection was found
		 */
		bool intersect(const HRay &ray, float *tHit, float *rayEpsilon, DifferentialGeometry *gd) const;
		/* if ray intersects.
		 * @ray ray to be intersected (in world space)
		 *
		 * Predicate that determines if an intersection occurs.
		 *
		 * @return true if an intersection was found.
		 */
		bool intersectP(const HRay &ray) const;
		float surfaceArea() const;

		const ponos::Transform *objectToWorld, *worldToObject;
		const bool reverseOrientation, transformSwapsHandedness;
		const uint32_t shapeId;
		static uint32_t nextShapeId;
	};

} // helios namespace

#endif
