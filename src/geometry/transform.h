#pragma once

#include "geometry/matrix.h"
#include "geometry/normal.h"
#include "geometry/point.h"
#include "geometry/ray.h"
#include "geometry/vector.h"

namespace ponos {

  class Transform {
  public:
    Transform() {}
    Transform(const Matrix4x4& mat);
    Transform(const Matrix4x4& mat, const Matrix4x4 inv_mat);
    Transform(const float mat[4][4]);
    friend Transform inverse(const Transform& t);
    Point3 operator()(const Point3& p) const {
      float x = p.x, y = p.y, z = p.z;
      float xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
      float yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
      float zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
      float wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
      if (wp == 1.f) return Point3(xp, yp, zp);
      return Point3(xp, yp, zp) / wp;
    }
    void operator()(const Point3& p, Point3* r) const {
      float x = p.x, y = p.y, z = p.z;
      r->x = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
      r->y = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
      r->z = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
      float wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
      if (wp != 1.f) *r /= wp;
    }
    Vector3 operator()(const Vector3& v) const {
      float x  = v.x, y = v.y, z = v.z;
      return Vector3(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
                     m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
                     m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
    }
    void operator()(const Vector3& v, Vector3* r) const {
      float x  = v.x, y = v.y, z = v.z;
      r->x = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z;
      r->y = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z;
      r->z = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z;
    }
    Normal operator()(const Normal& n) const {
      float x = n.x, y = n.y, z = n.z;
      return Normal(m_inv.m[0][0] * x + m_inv.m[1][0] * y + m_inv.m[2][0] * z,
                    m_inv.m[0][1] * x + m_inv.m[1][1] * y + m_inv.m[2][1] * z,
                    m_inv.m[0][2] * x + m_inv.m[1][2] * y + m_inv.m[2][2] * z);
    }
    Ray operator()(const Ray& r) {
      Ray ret = r;
      (*this)(ret.o, &ret.o);
      (*this)(ret.d, &ret.d);
      return ret;
    }
    Transform operator*(const Transform& t) const {
      Matrix4x4 m1 = Matrix4x4::mul(m, t.m);
      Matrix4x4 m1_inv = Matrix4x4::mul(t.m_inv, m_inv);
      return Transform(m1, m1_inv);
    }
  private:
    Matrix4x4 m, m_inv;
  };

  Transform inverse(const Transform& t);
  Transform translate(const Vector3 &d);
  Transform scale(float x, float y, float z);
  Transform rotateX(float angle);
  Transform rotateY(float angle);
  Transform rotateZ(float angle);
  Transform rotate(float angle, const Vector3& axis);
  Transform lookAt(const Point3& pos, const Point3& target, const Vector3& up);

}; // ponos namespace
