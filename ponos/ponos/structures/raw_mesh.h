/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef PONOS_STRUCTURES_RAW_MESH_H
#define PONOS_STRUCTURES_RAW_MESH_H

#include <ponos/geometry/transform.h>

#include <memory>
#include <vector>

namespace ponos {

enum class GeometricPrimitiveType {
  POINTS,
  LINES,
  LINE_STRIP,
  LINE_LOOP,
  TRIANGLES,
  TRIANGLE_STRIP,
  TRIANGLE_FAN,
  QUADS,
  TETRAHEDRA,
  CUSTOM
};

///  \brief mesh structure
/// Stores the elements_ of the mesh in simple arrays. This class
/// is the one that actually stores the geometry, then other
/// objects in the system can just use its reference avoiding
/// duplicating data.
class RawMesh {
public:
  RawMesh() {
    positionDescriptor.elementSize = 3;
    primitiveType = GeometricPrimitiveType::TRIANGLES;
  }
  virtual ~RawMesh() = default;
  struct IndexData {
    int positionIndex;
    int normalIndex;
    int texcoordIndex;
  };
  /// Applies transform **t** to all vertices.
  /// \param t transform
  void apply(const Transform &t);
  void splitIndexData();
  void computeBBox();
  /// Append positions.
  /// \param l coordinate values {v0, v1, .... }
  void addPosition(std::initializer_list<float> l);
  /// Append normals.
  /// \param l coordinate values {n0, n1, .... }
  void addNormal(std::initializer_list<float> l);
  /// Append texture coordinates.
  /// \param l coordinate values {n0, n1, .... }
  void addUV(std::initializer_list<float> l);
  /// add face(s). Append indices.
  /// \param l indices values {v00, v01, .... }
  void addFace(std::initializer_list<IndexData> l);
  /// add face(s)
  /// \param l vertices indices list
  void addFace(std::initializer_list<int> l);
  /// \param i element index
  /// \returns bbox of element (object space).
  [[nodiscard]] bbox3 elementBBox(size_t i) const;
  /// \param e element index
  /// \param v position index (inside element) [0..**elementSize**]
  /// \return position **v** from element **e**.
  [[nodiscard]] point3 positionElement(size_t e, size_t v) const;
  /// Builds a single array with interleaved information for vertex buffer
  /// (vertex | normal | texcoords | ... )
  void buildInterleavedData();
  /// \brief orient faces. Rearrange faces vertices order to fix face's normal.
  /// \param ccw make ccw ?
  void orientFaces(bool ccw = true);
  struct ArrayDescriptor {
    size_t elementSize = 0; //!< number of components per element.
    size_t count = 0;       //!< number of elements_.
  };
  /// clears everything, sets zero to all fields
  void clear();
  ArrayDescriptor interleavedDescriptor; //!< interleaved data descriptor
  std::vector<float> interleavedData; //!< flat array on the form [Vi Ni Ti ...]

  ArrayDescriptor meshDescriptor;     //!< mesh description
  ArrayDescriptor positionDescriptor; //!< position description
  ArrayDescriptor texcoordDescriptor; //!< texture coordinates description
  ArrayDescriptor normalDescriptor;   //!<< normal description

  std::vector<IndexData> indices;

  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<float> texcoords;
  std::vector<uint> positionsIndices;
  std::vector<uint> normalsIndices;
  std::vector<uint> texcoordsIndices;
  bbox3 bbox; //!< bounding box in object space
  GeometricPrimitiveType primitiveType;
};

///  Reshapes the raw mesh to fit into a BBox.
/// \param rm **[in/out]** Raw Mesh that will soffer the transformation. Note:
/// Must have this Bounding Box computed before passsing to this function.
/// \param bbox **[in]** Bounding box to be fitted
void fitToBBox(RawMesh *rm, const bbox2 &bbox);

std::ostream &operator<<(std::ostream &os, RawMesh &rm);

typedef std::shared_ptr<RawMesh> RawMeshSPtr;
} // namespace ponos

#endif // PONOS_STRUCTURES_RAW_MESH_H
