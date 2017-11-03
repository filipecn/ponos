#ifndef PONOS_STRUCTURES_MESH_H
#define PONOS_STRUCTURES_MESH_H

#include <ponos/structures/raw_mesh.h>

#include <memory>

namespace ponos {

/* mesh
 * The pair of a Mesh and a transform.
 */
class Mesh {
public:
  /* Constructor.
   * @m **[in]**
   * @t **[in]**
   * @return
   */
  Mesh(const ponos::RawMesh *m, const ponos::Transform &t);
  virtual ~Mesh() {}

  bool intersect(const ponos::Point3 &p);
  const ponos::BBox &getBBox() const;
  const ponos::RawMesh *getMesh() const;
  const ponos::Transform &getTransform() const;

private:
  std::shared_ptr<const ponos::RawMesh> mesh;
  ponos::Transform transform;
  ponos::BBox bbox;
};

/* mesh
 * The pair of a Mesh and a transform.
 */
class Mesh2D {
public:
  /* Constructor.
   * @m **[in]**
   * @t **[in]**
   * @return
   */
  Mesh2D(const ponos::RawMesh *m, const ponos::Transform2D &t);
  virtual ~Mesh2D() {}

  bool intersect(const ponos::Point2 &p);
  const ponos::BBox2D &getBBox() const;
  const ponos::RawMesh *getMesh() const;
  const ponos::Transform2D &getTransform() const;

private:
  std::shared_ptr<const ponos::RawMesh> mesh;
  ponos::Transform2D transform;
  ponos::BBox2D bbox;
};

} // ponos namespace

#endif // AERGIA_SCENE_MESH_H
