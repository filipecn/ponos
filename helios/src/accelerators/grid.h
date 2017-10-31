#ifndef HELIOS_ACCELERATORS_GRID_H
#define HELIOS_ACCELERATORS_GRID_H

#include "accelerators/aggregate.h"

#include <ponos.h>
#include <memory>

namespace helios {
	class Voxel;

	/* Accelerator structure.
	 * GridAccel is an accelerator that divides an axis-aligned region of
	 * space into equal-sized box-shaped chunks. Each <Voxel> stores
	 * references to the primitives (<helios::Primitive>)  that overlap it.
	 */
	class GridAccel : public Aggregate {
  	public:
			/* GridAccel constructor.
			 * @p list of primitives (<helios::Primitive>) references
			 * @refineFirst if true, refine all primitives prior to grid creation
			 *
			 * The grid resolution is computed based on the number of primitives
			 */
	 		GridAccel(const std::vector<std::shared_ptr<Primitive> >& p, bool refineFirst);
			// default destructor
			virtual ~GridAccel() {}
			ponos::BBox worldBound() const override;
			/* @inherit */
			bool intersect(const HRay &r, Intersection *in) const override;
			bool intersectP(const HRay &r) const override;
			const AreaLight *getAreaLight() const override;
		private:
			/* .
			 * @P world space
			 * @axis 0~3
			 * @return the coordinate of the voxel that contain P
			 */
			int posToVoxel(const ponos::Point3& P, int axis) const;
			/* .
			 * @p index space
			 * @axis 0~3
			 * @return position of the voxel's lower corner
			 */
			float voxelToPos(int p, int axis) const;
			int offset(int x, int y, int z) const {
				return z * nVoxels[0] * nVoxels[1] + y * nVoxels[0] + x;
			}
			std::vector<std::shared_ptr<Primitive> > primitives;
			// grid resolution
			int nVoxels[3];
			ponos::BBox bounds;
			ponos::vec3 width, invWidth;
			std::vector<Voxel*> voxels;
			ponos::MemoryArena voxelArena;
			// read-writer mutex for grid
			// mutable std::mutex rwMutex;
	};

	/* A region in space.
	 * Voxel represents a region in space. It stores the
	 * primitives that overlap this region.
	 */
	struct Voxel {
		Voxel() { allCanIntersect = false; }
		Voxel(std::shared_ptr<Primitive> op) {
			allCanIntersect = false;
			primitives.emplace_back(op);
		}

		void addPrimitive(std::shared_ptr<Primitive> prim) {
			primitives.emplace_back(prim);
		}

		// intersect handles the details of calling <Primitivve>::<intersect>() methods
		bool intersect(const HRay &ray, Intersection *isect/*, RWMutexLock &lock*/) {
			// refine primitives in voxel if needed
			if(!allCanIntersect) {
				// lock.upgradeToWrite();
				for(uint32 i = 0; i < primitives.size(); i++) {
					std::shared_ptr<Primitive> &prim = primitives[i];
					// refine prim if it's not intersectable
					if(!prim->canIntersect()) {
						std::vector<std::shared_ptr<Primitive> > p;
						prim->fullyRefine(p);
						if(p.size() == 1)
							primitives[i] = p[0];
						else
							primitives[i].reset(new GridAccel(p, false));
					}
				}
				allCanIntersect = true;
				// lock.downgadeToRead();
			}
			// loop over primitives in voxel and find intersections
			bool hitSomething = false;
			for(auto &prim : primitives){
				if(prim->intersect(ray, isect))
					hitSomething = true;
			}
			return hitSomething;
		}
		bool intersectP(const HRay &ray/*, RWMutexLock &lock*/) {
			// refine primitives in voxel if needed
			if(!allCanIntersect) {
				// lock.upgradeToWrite();
				for(uint32 i = 0; i < primitives.size(); i++) {
					std::shared_ptr<Primitive> &prim = primitives[i];
					// refine prim if it's not intersectable
					if(!prim->canIntersect()) {
						std::vector<std::shared_ptr<Primitive> > p;
						prim->fullyRefine(p);
						if(p.size() == 1)
							primitives[i] = p[0];
						else
							primitives[i].reset(new GridAccel(p, false));
					}
				}
				allCanIntersect = true;
				// lock.downgadeToRead();
			}
			// loop over primitives in voxel and find intersections
			for(auto &prim : primitives){
				if(prim->intersectP(ray))
					return true;
			}
			return false;
		}
		private:
		std::vector<std::shared_ptr<Primitive> > primitives;
		// indicates whether all of the primitives in the voxel are known
		// to be intersectable.
		bool allCanIntersect;
	};

} // helios namespace

#endif

