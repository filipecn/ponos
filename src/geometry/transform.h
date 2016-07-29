#pragma once

#include "geometry/bbox.h"
#include "geometry/matrix.h"
#include "geometry/normal.h"
#include "geometry/point.h"
#include "geometry/ray.h"
#include "geometry/vector.h"
#include "log/debug.h"

namespace ponos {

  class Transform2D {
  public:
    Transform2D() {}
    Transform2D(const Matrix3x3& mat, const Matrix3x3 inv_mat);
    void reset();
    void translate(const Vector2 &d);
    void scale(float x, float y);
    friend Transform2D inverse(const Transform2D& t);
    Vector2 operator()(const Vector2& v) const {
      float x = v.x, y = v.y;
      return Vector2(m.m[0][0] * x + m.m[0][1] * y,
      m.m[1][0] * x + m.m[1][1] * y);
    }
    Point2 operator()(const Point2& p) const {
			float x = p.x, y = p.y;
				float xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2];
				float yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2];
				float wp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2];
				if (wp == 1.f) return Point2(xp, yp);
					return Point2(xp / wp, yp / wp);
		}
		Transform2D operator*(const Transform2D& t) const {
			Matrix3x3 m1 = Matrix3x3::mul(m, t.m);
			Matrix3x3 m1_inv = Matrix3x3::mul(t.m_inv, m_inv);
				return Transform2D(m1, m1_inv);
		}
		Vector2 getTranslate() const {
      return Vector2(m.m[0][2], m.m[1][2]);
    }
    Vector2 getScale() const {
      return s;
    }
    void computeInverse() {
      m_inv = inverse(m);
    }
  private:
    Matrix3x3 m, m_inv;
    Vector2 s;
  };

  Transform2D rotate(float angle);
  Transform2D inverse(const Transform2D& t);

  class Transform {
  public:
    Transform() {}
    Transform(const Matrix4x4& mat);
    Transform(const Matrix4x4& mat, const Matrix4x4 inv_mat);
    Transform(const float mat[4][4]);
    friend Transform inverse(const Transform& t);
    BBox operator()(const BBox &b) const {
      const Transform &M = *this;
      BBox ret(             M(Point3(b.pMin.x, b.pMin.y, b.pMin.z)));
      ret = make_union(ret, M(Point3(b.pMax.x, b.pMin.y, b.pMin.z)));
      ret = make_union(ret, M(Point3(b.pMin.x, b.pMax.y, b.pMin.z)));
      ret = make_union(ret, M(Point3(b.pMin.x, b.pMin.y, b.pMax.z)));
      ret = make_union(ret, M(Point3(b.pMin.x, b.pMax.y, b.pMax.z)));
      ret = make_union(ret, M(Point3(b.pMax.x, b.pMax.y, b.pMin.z)));
      ret = make_union(ret, M(Point3(b.pMax.x, b.pMin.y, b.pMax.z)));
      ret = make_union(ret, M(Point3(b.pMin.x, b.pMax.y, b.pMax.z)));
      return ret;
    }
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
    void operator()(const Ray& r, Ray* ret) const {
      (*this)(r.o, &ret->o);
      (*this)(r.d, &ret->d);
    }
    Transform operator*(const Transform& t) const {
      Matrix4x4 m1 = Matrix4x4::mul(m, t.m);
      Matrix4x4 m1_inv = Matrix4x4::mul(t.m_inv, m_inv);
      return Transform(m1, m1_inv);
    }
		Point3 operator*(const Point3& p) const {
			return Point3((transpose(m) * vec4(p.x, p.y, p.z, 1.f)).xyz());
		}
    bool swapsHandedness() const;
    const float* c_matrix() const {
      return &m.m[0][0];
    }
    const Matrix4x4& matrix() const {
      return m;
    }
    Vector3 getTranslate() {
      return Vector3(m.m[0][3], m.m[1][3], m.m[2][3]);
    }
    void computeInverse() {
      m_inv = inverse(m);
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
	Transform frustrum(float left, float right, float bottom, float top, float near, float far);
	Transform perspective(float fovy, float aspect, float zNear, float zFar);
  Transform lookAt(const Point3& pos, const Point3& target, const Vector3& up);
  Transform lookAtRH(const Point3& pos, const Point3& target, const Vector3& up);
  Transform ortho(float left, float right, float bottom, float top, float near = -1.f, float far = 1.f);

} // ponos namespace
