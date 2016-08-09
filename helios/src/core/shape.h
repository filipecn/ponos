#pragma once

#include "core/differential_geometry.h"
#include "geometry/h_ray.h"
#include <ponos.h>

#include <memory>
#include <vector>

namespace helios {

	class Shape {
		public:
		Shape(const ponos::Transform *o2w, const ponos::Transform *w2o, bool ro);

		virtual ponos::BBox objectBound() const;
		virtual void getShadingGeometry(const ponos::Transform &o2w,
				const DifferentialGeometry &dg,
				DifferentialGeometry *dgShading) const {
			*dgShading = dg;
		}

		ponos::BBox worldBound() const;
		bool canIntersect() const;
		void refine(std::vector<std::shared_ptr<Shape> > &refined) const;
		bool intersect(const HRay &ray, float *tHit, float *rayEpsilon, DifferentialGeometry *gd) const;
		bool intersectP(const HRay &ray) const;
		float surfaceArea() const;

		const ponos::Transform *objectToWorld, *worldToObject;
		const bool reverseOrientation, transformSwapsHandedness;
		const uint32_t shapeId;
		static uint32_t nextShapeId;
	};

} // helios namespace
