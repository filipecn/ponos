#include "core/primitive.h"

namespace helios {

	uint32_t Primitive::nextPrimitiveId = 1;

	void Primitive::fullyRefine(std::vector<std::shared_ptr<Primitive> > &refined) const {
		std::vector<std::shared_ptr<Primitive> > todo;
		todo.emplace_back(const_cast<Primitive*>(this));
		while(todo.size()) {
			std::shared_ptr<Primitive> p = todo.back();
			todo.pop_back();
			if(p->canIntersect())
				refined.emplace_back(p);
			else
				p->refine(todo);
		}
	}

	bool GeometricPrimitive::intersect(const HRay &r, Intersection *isect) const {
		float thit, rayEpsilon;
		if(!shape->intersect(r, &thit, &rayEpsilon, &isect->dg))
			return false;
		isect->primitive = this;
		isect->w2o = *shape->worldToObject;
		isect->o2w = *shape->objectToWorld;
		isect->shapeId = shape->shapeId;
		isect->primitiveId = primitiveId;
		isect->rayEpsilon = rayEpsilon;
		r.max_t = thit;
		return true;
	}


	bool TransformedPrimitive::intersect(const HRay &r, Intersection *isect) const {
		ponos::Transform worldToPrimitive;
		w2p.interpolate(r.time, &worldToPrimitive);
		ponos::Ray3 tr3 = worldToPrimitive(ponos::Ray3(r.o, r.d));
		HRay ray = r;
		ray.o = tr3.o;
		ray.d = tr3.d;
		if(!p->intersect(ray, isect))
			return false;
		r.max_t = ray.max_t;
		isect->primitiveId = primitiveId;
		if(!worldToPrimitive.isIdentity()) {
			// compute w2o transformation for instance
			isect->w2o = isect->w2o * worldToPrimitive;
			isect->o2w = ponos::inverse(isect->w2o);
			// transform instance's differential geometry to world space
			ponos::Transform p2w = ponos::inverse(worldToPrimitive);
			isect->dg.p = p2w(isect->dg.p);
			// TODO rest of page 191
			isect->dg.p = p2w(isect->dg.p);
			isect->dg.p = p2w(isect->dg.p);
			isect->dg.p = p2w(isect->dg.p);
			isect->dg.p = p2w(isect->dg.p);
			isect->dg.p = p2w(isect->dg.p);
		}
		return true;
	}
} // helios namespace
