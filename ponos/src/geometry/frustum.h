#pragma once

#include "geometry/plane.h"
#include "geometry/point.h"
#include "geometry/transform.h"
#include "geometry/vector.h"

namespace ponos {

	class Frustum {
  	public:
	 		Frustum() {}

			// http://gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf
			// it t is the projection matrix, then frustum is in view space
			// it t is the M * V * P matrix, then frustum is in model space
			void set(Transform t) {
				left.normal.x = t.matrix().m[3][0] + t.matrix().m[0][0];
				left.normal.y = t.matrix().m[3][1] + t.matrix().m[0][1];
				left.normal.z = t.matrix().m[3][2] + t.matrix().m[0][2];
				left.offset   = t.matrix().m[3][3] + t.matrix().m[0][3];

				right.normal.x = t.matrix().m[3][0] - t.matrix().m[0][0];
				right.normal.y = t.matrix().m[3][1] - t.matrix().m[0][1];
				right.normal.z = t.matrix().m[3][2] - t.matrix().m[0][2];
				right.offset   = t.matrix().m[3][3] - t.matrix().m[0][3];

				bottom.normal.x = t.matrix().m[3][0] + t.matrix().m[1][0];
				bottom.normal.y = t.matrix().m[3][1] + t.matrix().m[1][1];
				bottom.normal.z = t.matrix().m[3][2] + t.matrix().m[1][2];
				bottom.offset   = t.matrix().m[3][3] + t.matrix().m[1][3];

				top.normal.x = t.matrix().m[3][0] - t.matrix().m[1][0];
				top.normal.y = t.matrix().m[3][1] - t.matrix().m[1][1];
				top.normal.z = t.matrix().m[3][2] - t.matrix().m[1][2];
				top.offset   = t.matrix().m[3][3] - t.matrix().m[1][3];

				near.normal.x = t.matrix().m[3][0] + t.matrix().m[2][0];
				near.normal.y = t.matrix().m[3][1] + t.matrix().m[2][1];
				near.normal.z = t.matrix().m[3][2] + t.matrix().m[2][2];
				near.offset   = t.matrix().m[3][3] + t.matrix().m[2][3];

				far.normal.x = t.matrix().m[3][0] - t.matrix().m[2][0];
				far.normal.y = t.matrix().m[3][1] - t.matrix().m[2][1];
				far.normal.z = t.matrix().m[3][2] - t.matrix().m[2][2];
				far.offset   = t.matrix().m[3][3] - t.matrix().m[2][3];
			}

			Plane near;
			Plane far;
			Plane left;
			Plane right;
			Plane bottom;
			Plane top;
	};

} // ponos namespace

