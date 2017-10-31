#ifndef HELIOS_GEOMETRY_ANIMATED_TRANSFORM_H
#define HELIOS_GEOMETRY_ANIMATED_TRANSFORM_H

#include "geometry/h_ray.h"
#include <ponos.h>

namespace helios {

	class AnimatedTransform {
		public:
			AnimatedTransform(const ponos::Transform *t1, float time1, const ponos::Transform *t2, float time2)
				: startTime(time1), endTime(time2),
				startTransform(t1), endTransform(t2),
				actuallyAnimated(*startTransform != *endTransform) {
					decompose(startTransform->matrix(), &T[0], &R[0], &S[0]);
					decompose(endTransform->matrix(), &T[1], &R[1], &S[1]);
				}

			void decompose(const ponos::mat4 &m, ponos::vec3 *T, ponos::Quaternion *Rquat, ponos::mat4 *s) {
				// extract translation T from transformation matrix
				T->x = m.m[0][3];
				T->y = m.m[1][3];
				T->z = m.m[2][3];
				// compute new transformation matrix M without translation
				ponos::mat4 M = m;
				for(int i = 0; i < 3; i++)
					M.m[i][3] = M.m[3][i] = 0.f;
				M.m[3][3] = 1.f;
				ponos::mat4 r;
				ponos::decompose(M, r, *s);
				*Rquat = ponos::Quaternion(r);
			}
			void interpolate(float time, ponos::Transform *t) const {
				// handle boundary conditions for matrix interpolation
				// float dt = (time - startTime) / (endTime - startTime);
				// interpolate translation at dt
				// interpolate rotation at dt
				// interpolate scale at dt
				// compute interpolated matrix as product of interpolated components
			}

			void operator()(const HRay& r, HRay* ret) const {
				// TODO
			}
			virtual ~AnimatedTransform() {}

		private:
			const float startTime, endTime;
			const ponos::Transform *startTransform, *endTransform;
			const bool actuallyAnimated;
			ponos::vec3 T[2];
			ponos::Quaternion R[2];
			ponos::mat4 S[2];
	};

} // helios namespace

#endif // HELIOS_GEOMETRY_ANIMATED_TRANSFORM_H
