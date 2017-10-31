#include "accelerators/grid.h"

namespace helios {
	GridAccel::GridAccel(const std::vector<std::shared_ptr<Primitive> >& p, bool refineFirst) {
		// initialize primitives with primitives for grid
		if(refineFirst)
			for(auto primitive : p)
				primitive->fullyRefine(primitives);
		else
			primitives = p;
		// compute bounds and choose grid resolution
		for(size_t i = 0; i < primitives.size(); i++)
			bounds = ponos::make_union(bounds, primitives[i]->worldBound());
		ponos::vec3 delta = bounds.pMax - bounds.pMin;
		// find voxels per unit dist
		int maxAxis = bounds.maxExtent();
		float invMaxWidth = 1.f / delta[maxAxis];
		float cubeRoot = 3.f * powf(static_cast<float>(primitives.size()), 1.f / 3.f);
		float voxelsPerUnitDist = cubeRoot * invMaxWidth;
		for(int axis = 0; axis < 3; axis++) {
			nVoxels[axis] = ponos::round2Int(delta[axis] * voxelsPerUnitDist);
			nVoxels[axis] = ponos::clamp(nVoxels[axis], 1, 64);
		}
		// compute voxel widths and allocate voxels
		for(int axis = 0; axis < 3; axis++) {
			width[axis] = delta[axis] / nVoxels[axis];
			invWidth[axis] = (width[axis] == 0.f) ? 0.f : 1.f / width[axis];
		}
		int nv = nVoxels[0] * nVoxels[1] * nVoxels[2];
		voxels.resize(nv, nullptr);
		// add primitives to grid voxels
		for(size_t i = 0; i < primitives.size(); i++) {
			// find voxel extent to primitive
			ponos::BBox pb = primitives[i]->worldBound();
			int vmin[3], vmax[3];
			for(int axis = 0; axis < 3; axis++) {
				vmin[axis] = posToVoxel(pb.pMin, axis);
				vmax[axis] = posToVoxel(pb.pMax, axis);
			}
			// add primitive to overlaping voxels
			for(int z = vmin[2]; z <= vmax[2]; z++)
				for(int y = vmin[1]; y <= vmax[1]; y++)
					for(int x = vmin[0]; x <= vmax[0]; x++) {
						int o = offset(x, y, z);
						if(!voxels[o]) {
							voxels[o] = voxelArena.alloc<Voxel>();
							*voxels[o] = Voxel(primitives[i]);
						} else {
							voxels[o]->addPrimitive(primitives[i]);
						}
					}
		}
		// creat reader-writer mutex for grid
	}

	int GridAccel::posToVoxel(const ponos::Point3& P, int axis) const {
		int v = ponos::round2Int((P[axis] - bounds.pMin[axis]) * invWidth[axis]);
		return ponos::clamp(v, 0, nVoxels[axis] - 1);
	}

	float GridAccel::voxelToPos(int p, int axis) const {
		return bounds.pMin[axis] + p * width[axis];
	}

	ponos::BBox GridAccel::worldBound() const {
		return bounds;
	}

	bool GridAccel::intersect(const HRay &r, Intersection *in) const {
		// check ray against overall grid bounds
		float rayT;
		if(bounds.inside(r(r.min_t)))
			rayT = r.min_t;
		else if(!ponos::bbox_ray_intersection(bounds, ponos::Ray3(r.o, r.d), rayT))
			return false;
		ponos::Point3 gridIntersect = r(rayT);
		// set up 3D DDA for ray
		float nextCrossingT[3], deltaT[3];
		int step[3], out[3], pos[3];
		for(int axis = 0; axis < 3; axis++) {
			// compute current voxel for axis
			pos[axis] = posToVoxel(gridIntersect, axis);
			if(r.d[axis] >= 0) {
				// handle ray with positive direction for voxel stepping
				nextCrossingT[axis] = rayT +
					(voxelToPos(pos[axis] + 1, axis) - gridIntersect[axis]) / r.d[axis];
				deltaT[axis] = width[axis] / r.d[axis];
				step[axis] = 1;
				out[axis] = nVoxels[axis];
			}
			else {
				// handle ray with negative direction for voxel stepping
				nextCrossingT[axis] = rayT +
					(voxelToPos(pos[axis], axis) - gridIntersect[axis]) / r.d[axis];
				deltaT[axis] = -width[axis] / r.d[axis];
				step[axis] = -1;
				out[axis] = -nVoxels[axis];
			}
		}
		// walk ray through voxel grid
		// RWMutexLock lock(*rwMutex, READ);
		bool hitSomething = false;
		for(;;) {
			// check for intersection in current voxel
			Voxel *voxel = voxels[offset(pos[0], pos[1], pos[2])];
			if(voxel != nullptr)
				hitSomething |= voxel->intersect(r, in/*, lock*/);
			// advance to next voxel
			// find stepAxis for stepping to next voxel
			int bits = ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
								 ((nextCrossingT[0] < nextCrossingT[2]) << 1) +
								 ((nextCrossingT[1] < nextCrossingT[2]));
			const int compToAxis[8] = { 2, 1, 2, 1, 2, 2, 0, 0 };
			int stepAxis = compToAxis[bits];
			if(r.max_t < nextCrossingT[stepAxis])
				break;
			pos[stepAxis] += step[stepAxis];
			if(pos[stepAxis] == out[stepAxis])
				break;
			nextCrossingT[stepAxis] += deltaT[stepAxis];
		}
		return hitSomething;
	}

	bool GridAccel::intersectP(const HRay &r) const {
		// check ray against overall grid bounds
		float rayT;
		if(bounds.inside(r(r.min_t)))
			rayT = r.min_t;
		else if(!ponos::bbox_ray_intersection(bounds, ponos::Ray3(r.o, r.d), rayT))
			return false;
		ponos::Point3 gridIntersect = r(rayT);
		// set up 3D DDA for ray
		float nextCrossingT[3], deltaT[3];
		int step[3], out[3], pos[3];
		for(int axis = 0; axis < 3; axis++) {
			// compute current voxel for axis
			pos[axis] = posToVoxel(gridIntersect, axis);
			if(r.d[axis] >= 0) {
				// handle ray with positive direction for voxel stepping
				nextCrossingT[axis] = rayT +
					(voxelToPos(pos[axis] + 1, axis) - gridIntersect[axis]) / r.d[axis];
				deltaT[axis] = width[axis] / r.d[axis];
				step[axis] = 1;
				out[axis] = nVoxels[axis];
			}
			else {
				// handle ray with negative direction for voxel stepping
				nextCrossingT[axis] = rayT +
					(voxelToPos(pos[axis], axis) - gridIntersect[axis]) / r.d[axis];
				deltaT[axis] = -width[axis] / r.d[axis];
				step[axis] = -1;
				out[axis] = -nVoxels[axis];
			}
		}
		// walk ray through voxel grid
		// RWMutexLock lock(*rwMutex, READ);
		for(;;) {
			// check for intersection in current voxel
			Voxel *voxel = voxels[offset(pos[0], pos[1], pos[2])];
			if(voxel != nullptr && voxel->intersectP(r/*, lock*/))
				return true;
			// advance to next voxel
			// find stepAxis for stepping to next voxel
			int bits = ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
								 ((nextCrossingT[0] < nextCrossingT[2]) << 1) +
								 ((nextCrossingT[1] < nextCrossingT[2]));
			const int compToAxis[8] = { 2, 1, 2, 1, 2, 2, 0, 0 };
			int stepAxis = compToAxis[bits];
			if(r.max_t < nextCrossingT[stepAxis])
				break;
			pos[stepAxis] += step[stepAxis];
			if(pos[stepAxis] == out[stepAxis])
				break;
			nextCrossingT[stepAxis] += deltaT[stepAxis];
		}
		return false;
	}

	const AreaLight *GridAccel::getAreaLight() const {
		// TODO
		return nullptr;
	}

}
