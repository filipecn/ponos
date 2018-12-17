#ifndef HELIOS_CORE_PRIMITIVE_H
#define HELIOS_CORE_PRIMITIVE_H

#include "core/bsdf.h"
#include "core/differential_geometry.h"
#include "core/intersection.h"
#include "core/light.h"
#include "core/material.h"
#include "core/shape.h"
#include "geometry/animated_transform.h"
#include "geometry/h_ray.h"

#include <ponos.h>

#include <memory>
#include <vector>

namespace helios {

	/* Scene element.
	 * A primitive is a simple scene element which can interact with light
	 * and compute intersections with other elements.
	 */
	class Primitive {
		public:
			Primitive()
				: primitiveId(nextPrimitiveId++) {}
			virtual ~Primitive() {}

			virtual ponos::BBox worldBound() const = 0;
			// intersection
			virtual bool canIntersect() const;
			/* as
			 * @r ray to be intersected
			 * @in intersection
			 *
			 * asd
			 *
			 * @return true if intersection exists
			 */
			virtual bool intersect(const HRay &r, Intersection *in) const = 0;
			virtual bool intersectP(const HRay &r) const = 0;
			virtual void refine(std::vector<std::shared_ptr<Primitive> >& refined) const;
			void fullyRefine(std::vector<std::shared_ptr<Primitive> > &refined) const;
			// material
			virtual const AreaLight *getAreaLight() const = 0;
      //virtual BSDF *getBSDF(const DifferentialGeometry &gd, const ponos::Transform &o2w, ponos::MemoryArena &arena) const = 0;
			// virtual BSSRDF *getBSSRDF(const DifferentialGeometry &gd, const Transform &o2w, MemoryArena &arena) const = 0;
			const uint32_t primitiveId;
			static uint32_t nextPrimitiveId;
	};

	class GeometricPrimitive : public Primitive {
		public:
			GeometricPrimitive(const std::shared_ptr<Shape> &s, const std::shared_ptr<Material> &m, AreaLight *a)
				: shape(s),
				material(m),
				areaLight(a) {}
			// intersection
			virtual bool canIntersect() const override;
			bool intersect(const HRay &r, Intersection *isect) const override;
			// bool intersectP(const HRay &r) const override;
			// void refine(std::vector<std::shared_ptr<Primitive> >& refined) const override;

			// TODO page 189
		private:
			std::shared_ptr<Shape> shape;
			std::shared_ptr<Material> material;
			AreaLight *areaLight;
	};

	class TransformedPrimitive : public Primitive {
		public:
			TransformedPrimitive(std::shared_ptr<Primitive> &_p, const AnimatedTransform &_w2p)
				: p(_p), w2p(_w2p) {}

			bool intersect(const HRay &r, Intersection *isect) const override;
		private:
			std::shared_ptr<Primitive> p;
			const AnimatedTransform w2p;
	};

} // helios namespace

#endif
