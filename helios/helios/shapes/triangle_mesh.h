#ifndef HELIOS_SHAPES_TRIANGLE_MESH_H
#define HELIOS_SHAPES_TRIANGLE_MESH_H

#include <helios/core/shape.h>
#include <helios/core/texture.h>

#include <ponos.h>

#include <memory>
#include <vector>

namespace helios {

class TriangleMesh;

/* Triangle from <TriangleMesh>.
 *
 * Stores a pointer to the parent <TriangleMesh> that it came from and a
 * pointer to its 3 vertex indices in the mesh.
 */
class Triangle : public Shape {
public:
  /* Constructor.
   * @o2w object to world transformation
   * @w2o world to object transformation
   * @ro reverse ortientation: indicates if the surface normal directions
   * should be reversed from default (default = normals pointing outside)
   * @m parent <TriangleMesh> pointer
   * @n first vertex index
   */
  Triangle(const ponos::Transform *o2w, const ponos::Transform *w2o, bool ro,
           const TriangleMesh *m, uint32 n);

  /* @inherit */
  ponos::BBox objectBound() const override;
  /* @inherit */
  ponos::BBox worldBound() const;
  /* @inherit */
  bool canIntersect() const { return false; }
  /* @inherit */
  bool intersect(const HRay &ray, float *tHit, float *rayEpsilon,
                 DifferentialGeometry *dg) const;
  /* @inherit */
  virtual void
  getShadingGeometry(const ponos::Transform &o2w,
                     const DifferentialGeometry &dg,
                     DifferentialGeometry *dgShading) const override;
  float surfaceArea() const;

private:
  void getUVs(ponos::Point2 uvs[3]) const;

  std::shared_ptr<const TriangleMesh> mesh;
  const uint32 *v;
};

/* Stores a number of triangles together.
 *
 * Different from other shapes, the positions are stored in world space,
 * avoiding transforming rays many times during rendering.
 * However normals and tangents are stored in object space.
 */
class TriangleMesh : public Shape {
public:
  /* Constructor.
   * @o2w object to world transformation
   * @w2o world to object transformation
   * @ro reverse ortientation: indicates if the surface normal directions
   * should be reversed from default (default = normals pointing outside)
   * @v array of vertices
   * @i array of indices, 3 per triangle. (if n triangles, array size is 3n)
   * @n [optional] array of normals, one per vertex
   * @s [optional] array of tangents, one per vertex
   * @uv [optional] array of uv coordinates, one uv per vertex
   * @atex [optional] alpha mask texture
   */
  TriangleMesh(const ponos::Transform *o2w, const ponos::Transform *w2o,
               bool ro, const std::vector<ponos::Point3> &v,
               const std::vector<uint32> &i,
               const std::vector<ponos::Normal> &n,
               const std::vector<ponos::vec3> &s,
               const std::vector<ponos::Point2> &uv,
               const std::shared_ptr<Texture<float>> &atex);

  /* @inherit */
  ponos::BBox objectBound() const override;
  /* @inherit */
  ponos::BBox worldBound() const;
  /* @inherit */
  bool canIntersect() const { return false; }
  /* @inherit */
  void refine(std::vector<std::shared_ptr<Shape>> &refined) const;
  virtual ~TriangleMesh() {}

  friend class Triangle;

protected:
  std::vector<ponos::Point3> vertices;
  std::vector<uint32> indices;
  int ntrigs, nverts;
  std::vector<ponos::Normal> normals;
  std::vector<ponos::vec3> tangents;
  std::vector<ponos::Point2> uvs;
  std::shared_ptr<Texture<float>> alphaTexture;
};

} // namespace helios

#endif // HELIOS_SHAPES_TRIANGLE_MESH_H
