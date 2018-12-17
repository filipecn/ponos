// Created by filipecn on 2018-12-12.

#include "h_transform.h"
#include <helios/common/e_float.h>

namespace helios {

template <typename T>
ponos::Point3<T> HTransform::operator()(const ponos::Point3<T> &p,
                                        ponos::Vector3<T> *pError) const {
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
  *pError = gammaBound(3) * ponos::Vector3<T>(xAbsSum, yAbsSum, zAbsSum);
  if (wp == 1)
    return ponos::Point3<T>(xp, yp, zp);
  return ponos::Point3<T>(xp, yp, zp) / wp;
}

template <typename T>
ponos::Point3<T> HTransform::operator()(const ponos::Point3<T> &p,
                                        const ponos::Vector3<T> &pError,
                                        ponos::Vector3<T> *pTError) const {
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
  *pTError =
      (gammaBound(3) + 1) * ponos::Vector3<T>(dXAbsSum, dYAbsSum, dZAbsSum) +
      gammaBound(3) * ponos::Vector3<T>(xAbsSum, yAbsSum, zAbsSum);
  if (wp == 1)
    return ponos::Point3<T>(xp, yp, zp);
  return ponos::Point3<T>(xp, yp, zp) / wp;
}

template <typename T>
ponos::Vector3<T> HTransform::operator()(const ponos::Vector3<T> &v,
                                         ponos::Vector3<T> *vError) const {
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
  *vError = gammaBound(3) * ponos::Vector3<T>(xAbsSum, yAbsSum, zAbsSum);
  return ponos::Vector3<T>(xv, yv, zv);
}

template <typename T>
ponos::Vector3<T> HTransform::operator()(const ponos::Vector3<T> &v,
                                         const ponos::Vector3<T> &vError,
                                         ponos::Vector3<T> *vTError) const {
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
  *vTError =
      (gammaBound(3) + 1) * ponos::Vector3<T>(dXAbsSum, dYAbsSum, dZAbsSum) +
      gammaBound(3) * ponos::Vector3<T>(xAbsSum, yAbsSum, zAbsSum);
  return ponos::Vector3<T>(xv, yv, zv);
}

SurfaceInteraction HTransform::operator()(const SurfaceInteraction &si) const {
  SurfaceInteraction s;
  // transform p and pError
  s.p = (*this)(si.p, si.pError, &s.pError);
  // TODO: transform remaining members
  return s;
}

HRay HTransform::operator()(const HRay &r) const {
  ponos::vec3f oError;
  ponos::point3f o = (*this)(r.o, &oError);
  ponos::vec3f d = (*reinterpret_cast<const ponos::Transform *>(this))(r.d);
  // offset ray origin to edge of error bounds and compute max_t
  real_t max_t;
  return HRay(o, d, max_t, r.time /*, r.medium*/);
}

HRay HTransform::operator()(const HRay &r, ponos::vec3f *oError,
                            ponos::vec3f *dError) const {
  return HRay((*this)(r.o, oError), (*this)(r.d, dError), r.max_t, r.time);
}

bounds3f HTransform::operator()(const bounds3f &b) const {
  const Transform &M = *reinterpret_cast<const ponos::Transform *>(this);
  bounds3f ret(M(ponos::point3f(b.lower.x, b.lower.y, b.lower.z)));
  ret = make_union(ret, M(ponos::point3f(b.upper.x, b.lower.y, b.lower.z)));
  ret = make_union(ret, M(ponos::point3f(b.lower.x, b.upper.y, b.lower.z)));
  ret = make_union(ret, M(ponos::point3f(b.lower.x, b.lower.y, b.upper.z)));
  ret = make_union(ret, M(ponos::point3f(b.lower.x, b.upper.y, b.upper.z)));
  ret = make_union(ret, M(ponos::point3f(b.upper.x, b.upper.y, b.lower.z)));
  ret = make_union(ret, M(ponos::point3f(b.upper.x, b.lower.y, b.upper.z)));
  ret = make_union(ret, M(ponos::point3f(b.lower.x, b.upper.y, b.upper.z)));
  return ret;
}

} // namespace helios
