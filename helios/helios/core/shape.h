#ifndef HELIOS_CORE_SHAPE_H
#define HELIOS_CORE_SHAPE_H

#include <helios/core/interaction.h>
#include <helios/geometry/bounds.h>
#include <helios/geometry/h_ray.h>
#include <ponos/geometry/transform.h>

#include <helios/geometry/h_transform.h>
#include <memory>
#include <vector>

namespace helios {

/// Geometric shape Interface.
/// Provides all the geometric information about the <helios::Primitive> such as
/// its surface area and bounding box. All shapes receive a unique id.
class Shape {
public:
  /// \param o2w object to world transformation
  /// \param w2o world to object transformation
  /// \param ro reverse ortientation: indicates if the surface normal directions
  /// should be reversed from default (default = normals pointing outside).
  Shape(const HTransform *o2w, const HTransform *w2o, bool ro);
  /// Shape bounding box.
  /// \return bounding box of the shape (in object space)
  virtual bounds3f objectBound() const = 0;
  /// Shape bounding box.
  /// \return bounding box of the shapet (in world space)
  bounds3f worldBound() const;
  /* if is intersectable.
   *
   * Indicates if the shape can compute <helios::HRay> intersections.
   */
  bool canIntersect() const;
  /* refine shape.
   *
   * Splits shape into a group of new shapes.
   */
  void refine(std::vector<std::shared_ptr<Shape>> &refined) const;
  /// \param ray ray to be intersected (in world space).
  /// \param tHit [out] if an intersection is found, **tHit receives the
  /// parametric distance along the ray, between (0, tMax), to the intersection
  /// point.
  /// \param isect [out] information about the geometry of the intersection
  /// point
  /// \param testAlphaTexture true if the can be cutted away using a
  /// texture
  /// \return **true** if an intersection was found
  virtual bool intersect(const HRay &ray, real_t *tHit,
                         SurfaceInteraction *isect,
                         bool testAlphaTexture = true) const = 0;
  /// Predicate that determines if an intersection occurs.
  /// \param ray ray to be intersected (in world space).
  /// \param testAlphaTexture true if the can be cutted away using a
  /// texture
  /// \return true if an intersection exists
  virtual bool intersectP(const HRay &ray, bool testAlphaTexture = true) const;
  /// \return object's surface area
  virtual real_t surfaceArea() const = 0;

  const HTransform *objectToWorld; //!< object space to world space transform
  const HTransform *worldToObject; //!< world space to object space transform
  const bool reverseOrientation;   //!< indicates whether surface normals should
                                   //!< be inverted from default
  const bool transformSwapsHandedness; //!< precomputed from **o2w** transform

  const uint32_t shapeId;
  static uint32_t nextShapeId;
};

} // namespace helios

#endif
