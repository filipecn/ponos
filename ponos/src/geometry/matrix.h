#ifndef PONOS_GEOMETRY_MATRIX_H
#define PONOS_GEOMETRY_MATRIX_H

#include "geometry/vector.h"

namespace ponos {

	class Matrix4x4 {
		public:
			Matrix4x4();
			Matrix4x4(float mat[4][4]);
			Matrix4x4(float m00, float m01, float m02, float m03,
					float m10, float m11, float m12, float m13,
					float m20, float m21, float m22, float m23,
					float m30, float m31, float m32, float m33);
			void row_major(float *a) const {
				int k = 0;
				for(int i = 0; i < 4; i++)
					for(int j = 0; j < 4; j++)
						a[k++] = m[i][j];
			}
			void column_major(float *a) const {
				int k = 0;
				for(int i = 0; i < 4; i++)
					for(int j = 0; j < 4; j++)
						a[k++] = m[j][i];
			}
			Matrix4x4 operator*(const Matrix4x4 &mat) {
				Matrix4x4 r;
				for (int i = 0; i < 4; ++i)
					for (int j = 0; j < 4; ++j)
						r.m[i][j] = m[i][0] * mat.m[0][j] +
							m[i][1] * mat.m[1][j] +
							m[i][2] * mat.m[2][j] +
							m[i][3] * mat.m[3][j];
				return r;
			}
			Vector4 operator*(const Vector4 &v) const {
				Vector4 r;
				for (int i = 0; i < 4; i++)
					for (int j = 0; j < 4; j++)
						r[i] += m[i][j] * v[j];
				return r;
			}
			bool operator==(const Matrix4x4 &_m) const {
				for(int i = 0; i < 4; i++)
					for(int j = 0; j < 4; j++)
						if(!IS_EQUAL(m[i][j], _m.m[i][j]))
							return false;
				return true;
			}
			bool operator!=(const Matrix4x4 &_m) const {
				return !(*this == _m);
			}
			bool isIdentity() {
				for(int i = 0; i < 4; i++)
					for(int j = 0; j < 4; j++)
						if(i != j && !IS_EQUAL(m[i][j], 0.f))
							return false;
						else if(i == j && ~IS_EQUAL(m[i][j], 1.f))
							return false;
				return true;
			}
			static Matrix4x4 mul(const Matrix4x4& m1, const Matrix4x4& m2) {
				Matrix4x4 r;
				for (int i = 0; i < 4; ++i)
					for (int j = 0; j < 4; ++j)
						r.m[i][j] = m1.m[i][0] * m2.m[0][j] +
							m1.m[i][1] * m2.m[1][j] +
							m1.m[i][2] * m2.m[2][j] +
							m1.m[i][3] * m2.m[3][j];
				return r;
			}
			friend std::ostream& operator<<(std::ostream& os, const Matrix4x4& m);
			float m[4][4];
	};

	Matrix4x4 transpose(const Matrix4x4& m);
	Matrix4x4 inverse(const Matrix4x4& m);
	void decompose(const Matrix4x4& m, Matrix4x4& r, Matrix4x4& s);
	typedef Matrix4x4 mat4;

	class Matrix3x3 {
		public:
			Matrix3x3();
			Matrix3x3(float m00, float m01, float m02,
					float m10, float m11, float m12,
					float m20, float m21, float m22);
			void setIdentity();
			static Matrix3x3 mul(const Matrix3x3& m1, const Matrix3x3& m2) {
				Matrix3x3 r;
				for (int i = 0; i < 3; ++i)
					for (int j = 0; j < 3; ++j)
						r.m[i][j] = m1.m[i][0] * m2.m[0][j] +
							m1.m[i][1] * m2.m[1][j] +
							m1.m[i][2] * m2.m[2][j];
				return r;
			}
			friend std::ostream& operator<<(std::ostream& os, const Matrix3x3& m);
			float m[3][3];
	};

	Matrix3x3 transpose(const Matrix3x3& m);
	Matrix3x3 inverse(const Matrix3x3& m);

	typedef Matrix3x3 mat3;

} // ponos namespace

#endif
