#pragma once

namespace ponos {

    class Matrix4x4 {
    public:
        Matrix4x4();
        Matrix4x4(float mat[4][4]);
        Matrix4x4(float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33);

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
        float m[4][4];
    };

    Matrix4x4 transpose(const Matrix4x4& m);
    Matrix4x4 inverse(const Matrix4x4& m);

    typedef Matrix4x4 mat4;

    class Matrix3x3 {
    public:
        Matrix3x3();
        void setIdentity();

        float m[3][3];
    };

    typedef Matrix3x3 mat3;

}; // ponos namespace
