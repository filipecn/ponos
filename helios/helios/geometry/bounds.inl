#include <ponos/common/defs.h>
template <typename T> Bounds3<T>::Bounds3() : ponos::BBox3<T>() {}

template <typename T>
Bounds3<T>::Bounds3(const ponos::Point3<T> &p) : ponos::BBox3<T>(p) {}

template <typename T>
Bounds3<T>::Bounds3(const ponos::Point3<T> &p1, const ponos::Point3<T> &p2)
    : ponos::BBox3<T>(p1, p2) {}

template <typename T>
ponos::Point3<T> Bounds3<T>::lerp(const ponos::point3f &t) const {
  return ponos::Point3<T>(ponos::lerp(t.x, this->lower.x, this->upper.x),
                          ponos::lerp(t.y, this->lower.y, this->upper.y),
                          ponos::lerp(t.z, this->lower.z, this->upper.z));
}

template <typename T>
void Bounds3<T>::boundingSphere(ponos::Point3<T> &center,
                                real_t &radius) const {
  center = (this->lower + this->upper) / 2;
  radius = this->contains(*center, *this)
               ? ponos::distance(*center, this->upper)
               : 0;
}

template <typename T>
bool Bounds3<T>::intersect(const HRay &ray, real_t *hit0, real_t *hit1) const {
  real_t t0 = 0, t1 = ray.max_t;
  for (int i = 0; i < 3; i++) {
    // update interval for vox slab
    real_t invRayDir = 1 / ray.d[i];
    real_t tNear = (this->lower[i] - ray.o[i]) * invRayDir;
    real_t tFar = (this->upper[i] - ray.o[i]) * invRayDir;
    // update parametric interval from slab
    if (tNear > tFar)
      std::swap(tNear, tFar);
    // update tFar to ensure robust ray-bounds intersection
    tFar *= 1 + 2 * gammaBound(3);
    t0 = tNear > 0 ? tNear : t0;
    t1 = tFar < 1 ? tFar : t1;
    if (t0 > t1)
      return false;
  }
  if (hit0)
    *hit0 = t0;
  if (hit1)
    *hit1 = t1;
}

template <typename T>
bool Bounds3<T>::intersectP(const HRay &ray, const ponos::vec3f &invDir,
                            const int *dirIsNeg) const {
  const Bounds3<float> &bounds = *this;
  // check for ray intersection against x and y slabs
  real_t tMin = (bounds[dirIsNeg[0]].x - ray.o.x) * invDir.x;
  real_t tMax = (bounds[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
  real_t tyMin = (bounds[dirIsNeg[1]].y - ray.o.y) * invDir.y;
  real_t tyMax = (bounds[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;
  // update tMax and tyMax to ensure robust bounds intersection
  if (tMin > tyMax || tyMin > tMax)
    return false;
  if (tyMin > tMin)
    tMin = tyMin;
  if (tyMax < tMax)
    tMax = tyMax;
  // check for ray intersection against z slab
  real_t tzMin = (bounds[dirIsNeg[2]].z - ray.o.z) * invDir.z;
  real_t tzMax = (bounds[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;
  if (tMin > tzMax || tzMin > tMax)
    return false;
  if (tzMin > tMin)
    tMin = tzMin;
  if (tzMax < tMax)
    tMax = tzMax;
  return (tMin < ray.max_t) && (tMax > 0);
}
