#include "scene/bvh.h"

#include <algorithm>
#include <vector>

namespace aergia {

	BVH::BVH(SceneMesh *m) {
		sceneMesh.reset(m);
		std::vector<BVHElement> buildData;
		for(size_t i = 0; i < sceneMesh->rawMesh->elementCount; i++)
			buildData.emplace_back(BVHElement(i, sceneMesh->transform(sceneMesh->rawMesh->elementBBox(i))));
		uint totalNodes = 0;
		orderedElements.reserve(sceneMesh->rawMesh->elementCount);
		BVHNode *root = recursiveBuild(buildData, 0, sceneMesh->rawMesh->elementCount, &totalNodes, orderedElements);
		nodes.resize(totalNodes);
		for(uint32_t i = 0; i < totalNodes; i++)
			new (&nodes[i]) LinearBVHNode;
		uint32_t offset = 0;
		flattenBVHTree(root, &offset);
	}

	BVH::BVHNode* BVH::recursiveBuild(std::vector<BVHElement>& buildData, uint32_t start, uint32_t end, uint32_t* totalNodes, std::vector<uint32_t>& orderedElements) {
		(*totalNodes)++;
		BVHNode* node = new BVHNode();
		ponos::BBox bbox;
		for(uint32_t i = start; i < end; ++i)
			bbox = ponos::make_union(bbox, buildData[i].bounds);
		// compute all bounds
		uint32_t nElements = end - start;
		if(nElements == 1) {
			// create leaf node
			uint32_t firstElementOffset = orderedElements.size();
			for(uint32_t i = start; i < end; i++) {
				uint32_t elementNum = buildData[i].ind;
				orderedElements.emplace_back(elementNum);
			}
			node->initLeaf(firstElementOffset, nElements, bbox);
		} else {
			// compute bound of primitives
			ponos::BBox centroidBounds;
			for(uint32_t i = start; i < end; i++)
				centroidBounds = ponos::make_union(centroidBounds, buildData[i].centroid);
			int dim = centroidBounds.maxExtent();
			// partition primitives
			uint32_t mid = (start + end) / 2;
			if(centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
				// create leaf
				uint32_t firstElementOffset = orderedElements.size();
				for(uint32_t i = start; i < end; i++) {
					uint32_t elementNum = buildData[i].ind;
					orderedElements.emplace_back(elementNum);
				}
				node->initLeaf(firstElementOffset, nElements, bbox);
				return node;
			}
			// partition into equally sized subsets
			std::nth_element(&buildData[start], &buildData[mid], &buildData[end-1] + 1,
					ComparePoints(dim));
			node->initInterior(dim,
					recursiveBuild(buildData, start, mid, totalNodes, orderedElements),
					recursiveBuild(buildData, mid,   end, totalNodes, orderedElements));
		}
		return node;
	}

	uint32_t BVH::flattenBVHTree(BVHNode* node, uint32_t *offset) {
		LinearBVHNode *linearNode = &nodes[*offset];
		linearNode->bounds = node->bounds;
		uint32_t myOffset = (*offset)++;
		if(node->nElements > 0) {
			linearNode->elementsOffset = node->firstElementOffset;
			linearNode->nElements = node->nElements;
		} else {
			linearNode->axis = node->splitAxis;
			linearNode->nElements = 0;
			flattenBVHTree(node->children[0], offset);
			linearNode->secondChildOffset = flattenBVHTree(node->children[1], offset);
		}
		return myOffset;
	}

	bool BVH::intersect(const ponos::Ray3 &r, float *t) const {
		if(!nodes.size()) return false;
		bool hit = false;
		ponos::Point3 origin = r.o;
		ponos::vec3 invDir(1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z);
		uint32_t dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
		uint32_t todoOffset = 0, nodeNum = 0;
		uint32_t todo[64];
		while(true) {
			const LinearBVHNode* node = &nodes[nodeNum];
			if(intersect(node->bounds, r, invDir, dirIsNeg)) {
				if(node->nElements > 0) {
					// intersect ray with primitives
					for(uint32_t i = 0; i < node->nElements; i++) {
						ponos::Point3 v0 = sceneMesh->rawMesh->vertexElement(orderedElements[node->elementsOffset + i], 0);
						ponos::Point3 v1 = sceneMesh->rawMesh->vertexElement(orderedElements[node->elementsOffset + i], 1);
						ponos::Point3 v2 = sceneMesh->rawMesh->vertexElement(orderedElements[node->elementsOffset + i], 2);
						if(ponos::triangle_ray_intersection(v0, v1, v2, r))
							hit = true;
					}
					if(todoOffset == 0)
						break;
					nodeNum = todo[--todoOffset];
				} else {
					if(dirIsNeg[node->axis]) {
						todo[todoOffset++] = nodeNum + 1;
						nodeNum = node->secondChildOffset;
					} else {
						todo[todoOffset++] = node->secondChildOffset;
						nodeNum++;
					}
				}
			} else {
				if(todoOffset == 0) break;
				nodeNum = todo[--todoOffset];
			}
		}
		return hit;
	}

	bool BVH::intersect(const ponos::BBox& bounds, const ponos::Ray3& ray, const ponos::vec3& invDir, const uint32_t dirIsNeg[3]) const {
		float tmin = (bounds[    dirIsNeg[0]].x - ray.o.x) * invDir.x;
		float tmax = (bounds[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
		float tymin = (bounds[    dirIsNeg[1]].y - ray.o.y) * invDir.y;
		float tymax = (bounds[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;
		if((tmin < tymax) || (tymin > tmax))
			return false;
		if(tymin > tmin) tmin = tymin;
		if(tymax < tmax) tmax = tymax;
		float tzmin = (bounds[    dirIsNeg[2]].z - ray.o.z) * invDir.z;
		float tzmax = (bounds[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;
		if((tmin < tzmax) || (tzmin > tmax))
			return false;
		if(tzmin > tmin) tmin = tzmin;
		if(tzmax < tmax) tmax = tzmax;
		return tmax > 0;
	}

} // aergia namespace"
