// Created by filipecn on 2018-12-12.

#include "h_transform.h"
#include <helios/common/e_float.h>

using namespace ponos;

namespace helios {

template <typename T>
Point3<T> HTransform::operator()(const Point3<T> &p, Vector3<T> *pError) const {
  T x = p.x, y = p.y, z = p.z;
  // compute transformed coordinates from point
  T xp = this->m.m[0][0] * x + this->m.m[0][1] * y + this->m.m[0][2] * z +
         this->m.m[0][3];
  T yp = this->m.m[1][0] * x + this->m.m[1][1] * y + this->m.m[1][2] * z +
         this->m.m[1][3];
  T zp = this->m.m[2][0] * x + this->m.m[2][1] * y + this->m.m[2][2] * z +
         this->m.m[2][3];
  T wp = this->m.m[3][0] * x + this->m.m[3][1] * y + this->m.m[3][2] * z +
         this->m.m[3][3];
  // compute absolute error for transformed point
  T xAbsSum = (std::abs(this->m.m[0][0] * x) + std::abs(this->m.m[0][1] * y) +
               std::abs(this->m.m[0][2] * z) + std::abs(this->m.m[0][3]));
  T yAbsSum = (std::abs(this->m.m[1][0] * x) + std::abs(this->m.m[1][1] * y) +
               std::abs(this->m.m[1][2] * z) + std::abs(this->m.m[1][3]));
  T zAbsSum = (std::abs(this->m.m[2][0] * x) + std::abs(this->m.m[2][1] * y) +
               std::abs(this->m.m[2][2] * z) + std::abs(this->m.m[2][3]));
  *pError = gammaBound(3) * Vector3<T>(xAbsSum, yAbsSum, zAbsSum);
  if (wp == 1)
    return Point3<T>(xp, yp, zp);
  return Point3<T>(xp, yp, zp) / wp;
}

template <typename T>
Point3<T> HTransform::operator()(const Point3<T> &p, const Vector3<T> &pError,
                                 Vector3<T> *pTError) const {
  T x = p.x, y = p.y, z = p.z;
  // compute transformed coordinates from point
  T xp = this->m.m[0][0] * x + this->m.m[0][1] * y + this->m.m[0][2] * z +
         this->m.m[0][3];
  T yp = this->m.m[1][0] * x + this->m.m[1][1] * y + this->m.m[1][2] * z +
         this->m.m[1][3];
  T zp = this->m.m[2][0] * x + this->m.m[2][1] * y + this->m.m[2][2] * z +
         this->m.m[2][3];
  T wp = this->m.m[3][0] * x + this->m.m[3][1] * y + this->m.m[3][2] * z +
         this->m.m[3][3];
  // compute absolute error for transformed point
  T xAbsSum = (std::abs(this->m.m[0][0] * x) + std::abs(this->m.m[0][1] * y) +
               std::abs(this->m.m[0][2] * z) + std::abs(this->m.m[0][3]));
  T yAbsSum = (std::abs(this->m.m[1][0] * x) + std::abs(this->m.m[1][1] * y) +
               std::abs(this->m.m[1][2] * z) + std::abs(this->m.m[1][3]));
  T zAbsSum = (std::abs(this->m.m[2][0] * x) + std::abs(this->m.m[2][1] * y) +
               std::abs(this->m.m[2][2] * z) + std::abs(this->m.m[2][3]));
  T dXAbsSum = std::abs(this->m.m[0][0]) * pError.x +
               std::abs(this->m.m[0][1]) * pError.y +
               std::abs(this->m.m[0][2]) * pError.z;
  T dYAbsSum = std::abs(this->m.m[1][0]) * pError.x +
               std::abs(this->m.m[1][1]) * pError.y +
               std::abs(this->m.m[1][2]) * pError.z;
  T dZAbsSum = std::abs(this->m.m[2][0]) * pError.x +
               std::abs(this->m.m[2][1]) * pError.y +
               std::abs(this->m.m[2][2]) * pError.z;
  *pTError = (gammaBound(3) + 1) * Vector3<T>(dXAbsSum, dYAbsSum, dZAbsSum) +
             gammaBound(3) * Vector3<T>(xAbsSum, yAbsSum, zAbsSum);
  if (wp == 1)
    return Point3<T>(xp, yp, zp);
  return Point3<T>(xp, yp, zp) / wp;
}

template <typename T>
Vector3<T> HTransform::operator()(const Vector3<T> &v,
                                  Vector3<T> *vError) const {
  T x = v.x, y = v.y, z = v.z;
  T xv = this->m.m[0][0] * x + this->m.m[0][1] * y + this->m.m[0][2] * z;
  T yv = this->m.m[1][0] * x + this->m.m[1][1] * y + this->m.m[1][2] * z;
  T zv = this->m.m[2][0] * x + this->m.m[2][1] * y + this->m.m[2][2] * z;
  T xAbsSum = (std::abs(this->m.m[0][0] * x) + std::abs(this->m.m[0][1] * y) +
               std::abs(this->m.m[0][2] * z));
  T yAbsSum = (std::abs(this->m.m[1][0] * x) + std::abs(this->m.m[1][1] * y) +
               std::abs(this->m.m[1][2] * z));
  T zAbsSum = (std::abs(this->m.m[2][0] * x) + std::abs(this->m.m[2][1] * y) +
               std::abs(this->m.m[2][2] * z));
  *vError = gammaBound(3) * Vector3<T>(xAbsSum, yAbsSum, zAbsSum);
  return Vector3<T>(xv, yv, zv);
}

template <typename T>
Vector3<T> HTransform::operator()(const Vector3<T> &v, const Vector3<T> &vError,
                                  Vector3<T> *vTError) const {
  T x = v.x, y = v.y, z = v.z;
  T xv = this->m.m[0][0] * x + this->m.m[0][1] * y + this->m.m[0][2] * z;
  T yv = this->m.m[1][0] * x + this->m.m[1][1] * y + this->m.m[1][2] * z;
  T zv = this->m.m[2][0] * x + this->m.m[2][1] * y + this->m.m[2][2] * z;
  T xAbsSum = (std::abs(this->m.m[0][0] * x) + std::abs(this->m.m[0][1] * y) +
               std::abs(this->m.m[0][2] * z));
  T yAbsSum = (std::abs(this->m.m[1][0] * x) + std::abs(this->m.m[1][1] * y) +
               std::abs(this->m.m[1][2] * z));
  T zAbsSum = (std::abs(this->m.m[2][0] * x) + std::abs(this->m.m[2][1] * y) +
               std::abs(this->m.m[2][2] * z));
  T dXAbsSum = std::abs(this->m.m[0][0]) * vError.x +
               std::abs(this->m.m[0][1]) * vError.y +
               std::abs(this->m.m[0][2]) * vError.z;
  T dYAbsSum = std::abs(this->m.m[1][0]) * vError.x +
               std::abs(this->m.m[1][1]) * vError.y +
               std::abs(this->m.m[1][2]) * vError.z;
  T dZAbsSum = std::abs(this->m.m[2][0]) * vError.x +
               std::abs(this->m.m[2][1]) * vError.y +
               std::abs(this->m.m[2][2]) * vError.z;
  *vTError = (gammaBound(3) + 1) * Vector3<T>(dXAbsSum, dYAbsSum, dZAbsSum) +
             gammaBound(3) * Vector3<T>(xAbsSum, yAbsSum, zAbsSum);
  return Vector3<T>(xv, yv, zv);
}

SurfaceInteraction HTransform::operator()(const SurfaceInteraction &si) const {
  SurfaceInteraction s;
  // transform p and pError
  s.p = (*this)(si.p, si.pError, &s.pError);
  // TODO: transform remaining members
  return s;
}

HRay HTransform::operator()(const HRay &r) const {
  vec3f oError;
  point3f o = (*this)(r.o, &oError);
  vec3f d = (*reinterpret_cast<const Transform *>(this))(r.d);
  // offset ray origin to edge of error bounds and compute max_t
  real_t lenghtSquared = d.length2();
  real_t max_t = r.max_t;
  if (lenghtSquared > 0) {
    real_t dt = dot(abs(d), oError) / lenghtSquared;
    o += d * dt;
    max_t -= dt;
  }
  return HRay(o, d, max_t, r.time /*TODO , r.medium*/);
}

HRay HTransform::operator()(const HRay &r, vec3f *oError, vec3f *dError) const {
  return HRay((*this)(r.o, oError), (*this)(r.d, dError), r.max_t, r.time);
}

bounds3f HTransform::operator()(const bounds3f &b) const {
  const Transform &M = *reinterpret_cast<const Transform *>(this);
  bounds3f ret(M(point3f(b.lower.x, b.lower.y, b.lower.z)));
  ret = make_union(ret, M(point3f(b.upper.x, b.lower.y, b.lower.z)));
  ret = make_union(ret, M(point3f(b.lower.x, b.upper.y, b.lower.z)));
  ret = make_union(ret, M(point3f(b.lower.x, b.lower.y, b.upper.z)));
  ret = make_union(ret, M(point3f(b.lower.x, b.upper.y, b.upper.z)));
  ret = make_union(ret, M(point3f(b.upper.x, b.upper.y, b.lower.z)));
  ret = make_union(ret, M(point3f(b.upper.x, b.lower.y, b.upper.z)));
  ret = make_union(ret, M(point3f(b.lower.x, b.upper.y, b.upper.z)));
  return ret;
}

} // namespace helios
