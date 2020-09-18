#include "triangle_mesh.h"

#include <cstring>

namespace helios {

	Triangle::Triangle(const ponos::Transform *o2w,
			const ponos::Transform *w2o,
			bool ro,
			const TriangleMesh *m,
			uint32 n)
		: Shape(o2w, w2o, ro) {
			mesh.reset(m);
			v = &mesh->indices[3 * n];
		}

	ponos::BBox Triangle::objectBound() const {
		const ponos::Point3& p1 = mesh->vertices[v[0]];
		const ponos::Point3& p2 = mesh->vertices[v[1]];
		const ponos::Point3& p3 = mesh->vertices[v[2]];
		return ponos::make_union(ponos::BBox((*worldToObject)(p1),
					(*worldToObject)(p2)),
				(*worldToObject)(p3));
	}

	ponos::BBox Triangle::worldBound() const {
		const ponos::Point3& p1 = mesh->vertices[v[0]];
		const ponos::Point3& p2 = mesh->vertices[v[1]];
		const ponos::Point3& p3 = mesh->vertices[v[2]];
		return ponos::make_union(ponos::BBox(p1, p2), p3);
	}

	bool Triangle::intersect(const HRay &ray, float *tHit, float *rayEpsilon, DifferentialGeometry *dg) const {
		const ponos::Point3& p1 = mesh->vertices[v[0]];
		const ponos::Point3& p2 = mesh->vertices[v[1]];
		const ponos::Point3& p3 = mesh->vertices[v[2]];
		float t, b1, b2;
		if(!ponos::triangle_ray_intersection(p1, p2, p3, static_cast<ponos::Ray3>(ray), &t, &b1, &b2))
			return false;
		if(t < ray.min_t || t > ray.max_t)
			return false;
		// Compute triangle partial derivatives
		ponos::vec3 dpdu, dpdv;
		ponos::vec3 dp1 = p1 - p3;
		ponos::vec3 dp2 = p2 - p3;
		ponos::Point2 uvs[3];
		getUVs(uvs);
		// Compute deltas for triangle partial derivatives
		ponos::mat2 m(uvs[0].x - uvs[2].x, uvs[0].y - uvs[2].y,
							    uvs[1].x - uvs[2].x, uvs[1].y - uvs[2].y);
		if(m.determinant() == 0.f) {
			// Handle zero determinant for triangle partial derivative matrix
			ponos::makeCoordinateSystem(ponos::normalize(ponos::cross(p2 - p1, p3 - p1)), &dpdu, &dpdv);
		}
		else {
			ponos::mat2 inv = ponos::inverse(m);
			dpdu = inv.m[0][0] * dp1 + inv.m[0][1] * dp2;
			dpdv = inv.m[1][0] * dp1 + inv.m[1][1] * dp2;
		}
		// Interpolate (u, v) triangle parametric coordinates
		float b0 = 1 - b1 - b2;
		float tu = b0 * uvs[0][0] + b1 * uvs[1][0] + b2 * uvs[2][0];
		float tv = b0 * uvs[0][1] + b1 * uvs[1][1] + b2 * uvs[2][1];
		// Test intersection against alpha texture
		if(mesh->alphaTexture) {
			DifferentialGeometry dgLocal(ray(t), dpdu, dpdv,
					ponos::Normal(0, 0, 0), ponos::Normal(0, 0, 0),
					tu, tv, this);
			// TODO
			//if(mesh->alphaTexture->evaluate(dgLocal) == 0.f)
				//return false;
		}
		// Fill in DifferentialGeometry from triangle hit
		*dg = DifferentialGeometry(ray(t), dpdu, dpdv,
				ponos::Normal(0, 0, 0), ponos::Normal(0, 0, 0),
				tu, tv, this);
		*tHit = t;
		*rayEpsilon = 1e-3f * *tHit;
		return true;
	}

	float Triangle::surfaceArea() const {
		const ponos::Point3& p1 = mesh->vertices[v[0]];
		const ponos::Point3& p2 = mesh->vertices[v[1]];
		const ponos::Point3& p3 = mesh->vertices[v[2]];
		return .5f * ponos::cross(p2 - p1, p3 - p1).length();
	}

	void Triangle::getShadingGeometry(const ponos::Transform &o2w,
			const DifferentialGeometry &dg,
			DifferentialGeometry *dgShading) const {
		if(!mesh->normals.size() && !mesh->tangents.size()) {
			*dgShading = dg;
			return;
		}
		// Initialize Triangle shading geometry with normals and tangents
		// TODO pg 146
		// compute barycentric coordinates for point
		// Use normals and tangents to compute shading tangents for triangle, ss and ts
		//ponos::Normal dndu, dndv;
		// Compute dn/du and dn/dv for triangle shading geometry
		//*dgShading = DifferentialGeometry(dg.p, ss, ts,
				//(*objectToWorld)(dndu), (*objectToWorld)(dndv),
				//dg.u, dg.v, dg.shape);
	}

	void Triangle::getUVs(ponos::Point2 uvs[3]) const {
		if(mesh->uvs.size()) {
			uvs[0] = mesh->uvs[v[0]];
			uvs[1] = mesh->uvs[v[1]];
			uvs[2] = mesh->uvs[v[2]];
		} else {
			uvs[0] = ponos::Point2(0, 0);
			uvs[1] = ponos::Point2(1, 0);
			uvs[2] = ponos::Point2(1, 1);
		}
	}

	TriangleMesh::TriangleMesh(const ponos::Transform *o2w,
			const ponos::Transform *w2o,
			bool ro,
			const std::vector<ponos::Point3>& v,
			const std::vector<uint32>& i,
			const std::vector<ponos::Normal>& n,
			const std::vector<ponos::vec3>& s,
			const std::vector<ponos::Point2>& uv,
			const std::shared_ptr<Texture<float> >& atex)
		: Shape(o2w, w2o, ro),
		alphaTexture(atex) {
			// copy triangles
			indices.resize(i.size());
			memcpy(&indices[0], &i[0], sizeof(uint32) * i.size());
			ntrigs = indices.size() / 3;
			// copy vertices
			vertices.resize(v.size());
			memcpy(&vertices[0], &v[0], sizeof(ponos::Point3) * v.size());
			nverts = vertices.size();
			// copy uv
			if(uv.size()) {
				uvs.resize(uv.size());
				memcpy(&uvs[0], &uv[0], sizeof(ponos::Point2) * uv.size());
			}
			// copy normals
			if(n.size()) {
				normals.resize(n.size());
				memcpy(&normals[0], &n[0], sizeof(ponos::Normal) * n.size());
			}
			// copy tangents
			if(s.size()) {
				tangents.resize(s.size());
				memcpy(&tangents[0], &s[0], sizeof(ponos::Vector3) * s.size());
			}
			// Transform mesh vertices to world space
			for(ponos::Point3& vertex : vertices)
				vertex = (*objectToWorld)(vertex);
		}

	ponos::BBox TriangleMesh::objectBound() const {
		ponos::BBox objectBounds;
		for(const ponos::Point3& v : vertices)
			objectBounds = ponos::make_union(objectBounds, (*worldToObject)(v));
		return objectBounds;
	}

	ponos::BBox TriangleMesh::worldBound() const {
		ponos::BBox worldBounds;
		for(const ponos::Point3& v : vertices)
			worldBounds = ponos::make_union(worldBounds, v);
		return worldBounds;
	}

	void TriangleMesh::refine(std::vector<std::shared_ptr<Shape> >& refined) const {
		for(int i = 0; i < ntrigs; i++)
			refined.emplace_back(new Triangle(objectToWorld, worldToObject, reverseOrientation, this, i));
	}

} // helios namespace
