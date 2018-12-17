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
  virtual ~Mesh() = default;

  bool intersect(const ponos::point3 &p);
  const ponos::bbox3 &getBBox() const;
  const ponos::RawMesh *getMesh() const;
  const ponos::Transform &getTransform() const;

private:
  std::shared_ptr<const ponos::RawMesh> mesh;
  ponos::Transform transform;
  ponos::bbox3 bbox;
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
  Mesh2D(const ponos::RawMesh *m, const ponos::Transform2 &t);
  virtual ~Mesh2D() = default;

  bool intersect(const ponos::point2 &p);
  const ponos::bbox2 &getBBox() const;
  const ponos::RawMesh *getMesh() const;
  const ponos::Transform2 &getTransform() const;

private:
  std::shared_ptr<const ponos::RawMesh> mesh;
  ponos::Transform2 transform;
  ponos::bbox2 bbox;
};

} // namespace ponos

#endif // AERGIA_SCENE_MESH_H
