#pragma once

#include "geometry/transform.h"
#include "geometry/vector.h"

namespace ponos {

	class Quaternion {
		public:
			Quaternion();
			Quaternion(Vector3 _v, float _w);
			Quaternion(const Transform &t);
			Quaternion(const Matrix4x4 &m);
			Quaternion operator+(const Quaternion& q) const {
				return Quaternion(v + q.v, w + q.w);
			}
			Quaternion& operator+=(const Quaternion& q) {
				v += q.v;
				w += q.w;
				return *this;
			}
			Quaternion operator-(const Quaternion& q) const {
				return Quaternion(v - q.v, w - q.w);
			}
			Quaternion& operator-=(const Quaternion& q) {
				v -= q.v;
				w -= q.w;
				return *this;
			}
			Quaternion operator/(float d) const {
				return Quaternion(v / d, w / d);
			}
			Quaternion operator*(float d) const {
				return Quaternion(v * d, w * d);
			}
			bool operator==(const Quaternion& q) {
				return IS_EQUAL(w, q.w) && v == q.v;
			}
			void fromAxisAndAngle(const Vector3& _v, float angle) {
				float theta = TO_RADIANS(angle / 2.f);
				v = _v * sinf(theta);
				w = cosf(theta);
			}
			void fromMatrix(const Matrix4x4 &m) {
				// extracted from Shoemake, 1991
				const size_t X = 0;
				const size_t Y = 1;
				const size_t Z = 2;
				const size_t W = 3;
				float tr, s;
				tr = m.m[X][X] + m.m[Y][Y] + m.m[Z][Z];
				if(tr >= 0.f) {
					s = sqrtf(tr + m.m[W][W]);
					w = s * .5f;
					s = .5f / s;
					v.x = (m.m[Z][Y] - m.m[Y][Z]) * s;
					v.y = (m.m[X][Z] - m.m[Z][X]) * s;
					v.z = (m.m[Y][X] - m.m[X][Y]) * s;
				} else {
					size_t h = X;
					if(m.m[Y][Y] > m.m[X][X])
						h = Y;
					if(m.m[Z][Z] > m.m[h][h])
						h = Z;
					switch(h) {
#define case_macro(i, j, k, I, J, K) \
						case I:\
									 s = sqrtf((m.m[I][I] - (m.m[J][J] + m.m[K][K])) + m.m[W][W]);\
						v.i = s * .5f;\
						s = .5f / s;\
						v.j = (m.m[I][J] + m.m[J][I]) * s;\
						v.k = (m.m[K][I] + m.m[I][K]) * s;\
						w =   (m.m[K][J] + m.m[J][K]) * s;\
						break;
						case_macro(x, y, z, X, Y, Z);
						case_macro(y, z, x, Y, Z, X);
						case_macro(z, x, y, Z, X, Y);
#undef case_macro
					}
				}
				if(m.m[W][W] != 1.0) {
					s = 1.0 / sqrtf(m.m[W][W]);
				w *= s; v *= s;
			}
		}

    Transform toTransform() const {
      float m[4][4];
      m[0][0] = 1.f - 2.f * (v.y * v.y + v.z * v.z);
      m[1][0] = 2.f * (v.x * v.y + v.z * w);
      m[2][0] = 2.f * (v.x * v.z - v.y * w);
      m[3][0] = 0.f;

      m[0][1] = 2.f * (v.x * v.y - v.z * w);
      m[1][1] = 1.f - 2.f * (v.x * v.x + v.z * v.z);
      m[2][1] = 2.f * (v.y * v.z + v.x * w);
      m[3][1] = 0.f;

      m[0][2] = 2.f * (v.x * v.z + v.y * w);
      m[1][2] = 2.f * (v.y * v.z - v.x * w);
      m[2][2] = 1.f - 2.f * (v.x * v.x + v.y * v.y);
      m[3][2] = 0.f;

      m[0][3] = 0.f;
      m[1][3] = 0.f;
      m[2][3] = 0.f;
      m[3][3] = 1.f;

      return Transform(m);
    }

		friend std::ostream& operator<<(std::ostream& os, const Quaternion& q) {
			os << "[quat]\n" << q.v << " w = " << q.w << std::endl;;
			return os;
		}

    Vector3 v;
    float w;
  };

	typedef Quaternion quat;

	inline Quaternion operator*(float f, const Quaternion& q) {
		return q * f;
	}

  inline float dot(const Quaternion& q1, const Quaternion& q2) {
    return dot(q1.v, q2.v) + q1.w * q2.w;
  }

  inline Quaternion normalize(const Quaternion& q) {
    return q / sqrtf(dot(q, q));
  }

	inline Quaternion slerp(float t, const Quaternion &q1, const Quaternion &q2) {
		float cosTheta = dot(q1, q2);
		if(cosTheta > .9995f)
			return normalize((1.f - t) * q1 + t * q2);
		float theta = acosf(clamp(cosTheta, -1.f, 1.f));
		float thetap = theta * t;
		Quaternion qperp = normalize(q2 - q1 * cosTheta);
		return q1 * cosf(thetap) + qperp * sinf(thetap);
	}

} // ponos namespace
