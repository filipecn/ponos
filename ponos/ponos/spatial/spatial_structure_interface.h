#ifndef PONOS_SPATIAL_STRUCTURE_INTERFACE_H
#define PONOS_SPATIAL_STRUCTURE_INTERFACE_H

#include <ponos/geometry/ray.h>

#include <functional>

namespace ponos {

/* interface
 * Defines an interface for spatial structures. Acceleration schemes for fast
 * spatial queries and geometric objects arrangement.
 */
template <typename ObjectType> class SpatialStructureInterface2D {
public:
  SpatialStructureInterface2D() {}
  virtual ~SpatialStructureInterface2D() {}
  /* add
   * @o **[in]** object
   * Add **o** to the structure
   */
  virtual size_t add(ObjectType *o) = 0;
  /* iterate
   * @f **[in]** function called for each object
   * Iterates thourgh all objects in the structure
   */
  virtual void iterate(std::function<void(const ObjectType *o)> f) const = 0;
  /* init
   * some structures need to be initialized after all objects have been added
   */
  virtual void init() {}
  /* query
   * @r **[in]** ray
   * @o **[out | optional]** closest intersected object from ray's origin
   * @t **[out | optional]** parametric coordinate of intersection
   *
   * Intersect Scene with a ray.
   *
   * @return **nullptr** if no intersection found
   */
  virtual ObjectType *intersect(const ponos::Ray2 &r,
                                float *t = nullptr) const {
    return nullptr;
  }
};

/* interface
 * Defines an interface for spatial structures. Acceleration schemes for fast
 * spatial queries and geometric objects arrangement.
 */
template <typename ObjectType> class SpatialStructureInterface {
public:
  SpatialStructureInterface() {}
  virtual ~SpatialStructureInterface() {}
  /* add
   * @o **[in]** object
   * Add **o** to the structure
   */
  virtual void add(ObjectType *o) = 0;
  /* iterate
   * @f **[in]** function called for each object
   * Iterates thourgh all objects in the structure
   */
  virtual void iterate(std::function<void(const ObjectType *o)> f) const = 0;
  virtual void iterate(std::function<void(ObjectType *o)> f) = 0;
  /* init
   * some structures need to be initialized after all objects have been added
   */
  virtual void init() {}
  /* query
   * @r **[in]** ray
   * @o **[out | optional]** closest intersected object from ray's origin
   * @t **[out | optional]** parametric coordinate of intersection
   *
   * Intersect Scene with a ray.
   *
   * @return **nullptr** if no intersection found
   */
  virtual ObjectType *intersect(const ponos::Ray3 &r,
                                float *t = nullptr) const {
    PONOS_UNUSED_VARIABLE(t);
    PONOS_UNUSED_VARIABLE(r);
    return nullptr;
  }
};

} // ponos namespace

#endif // PONOS_SPATIAL_STRUCTURE_INTERFACE_H
