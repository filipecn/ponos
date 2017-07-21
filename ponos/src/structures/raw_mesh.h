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

#include "geometry/bbox.h"
#include "geometry/transform.h"

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
  CUSTOM
};

/** \brief mesh structure
 *
 * Stores the elements of the mesh in simple arrays. This class
 * is the one that actually stores the geometry, then other
 * objects in the system can just use its reference avoiding
 * duplicating data.
 */
class RawMesh {
public:
  RawMesh() {
    vertexDescriptor.elementSize = 3;
    primitiveType = GeometricPrimitiveType::TRIANGLES;
  }
  virtual ~RawMesh() {}
  struct IndexData {
    int vertexIndex;
    int normalIndex;
    int texcoordIndex;
  };
  /** \brief set
   * \param t transform
   * Applies transform **t** to all vertices.
   */
  void apply(const Transform &t);
  void splitIndexData();
  void computeBBox();
  /** \brief add vertex
   * \param coordinate values {v00, v01, .... }
   * Append positions.
   */
  void addVertex(std::initializer_list<float> l);
  /** \brief add face(s)
   * \param indices values {v00, v01, .... }
   * Append indices.
   */
  void addFace(std::initializer_list<IndexData> l);
  /** \brief get
   * \param i element index
   * \returns bbox of element (object space).
   */
  ponos::BBox elementBBox(size_t i) const;
  /** \brief get
   * \param e element index
   * \param v vertex index (inside element) [0..**elementSize**]
   * \return vertex **v** from element **e**.
   */
  ponos::Point3 vertexElement(size_t e, size_t v) const;
  /** \brief set
   *
   * Builds a single array with interleaved information for vertex buffer
   *(vertex | normal | texcoords | ... )
   */
  void buildInterleavedData();
  /** \brief orient faces
   * \param ccw make ccw ?
   * Rearrange faces vertices order to fix face's normal.
   */
  void orientFaces(bool ccw = true);
  struct ArrayDescriptor {
    /** \brief
     * \param s element size
     * \param c element count
     */
    ArrayDescriptor(size_t s = 0, size_t c = 0) : elementSize(s), count(c) {}
    size_t elementSize; //!< number of components per element.
    size_t count;       //!< number of elements.
  };
  /** clears everything, sets zero to all fields
   */
  void clear();
  ArrayDescriptor interleavedDescriptor; //!< interleaved data descriptor
  std::vector<float> interleavedData; //!< flat array on the form [Vi Ni Ti ...]

  ArrayDescriptor meshDescriptor;     //!< mesh description
  ArrayDescriptor vertexDescriptor;   //!< vertex description
  ArrayDescriptor texcoordDescriptor; //!< texture coordinates description
  ArrayDescriptor normalDescriptor;   //!<< normal description

  std::vector<IndexData> indices;

  std::vector<float> vertices;
  std::vector<float> normals;
  std::vector<float> texcoords;
  std::vector<uint> verticesIndices;
  std::vector<uint> normalsIndices;
  std::vector<uint> texcoordsIndices;
  BBox bbox; //!< bounding box in object space
  GeometricPrimitiveType primitiveType;
};

} // ponos namespace

#endif // PONOS_STRUCTURES_RAW_MESH_H
