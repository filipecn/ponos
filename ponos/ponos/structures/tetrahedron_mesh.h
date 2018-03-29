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

#ifndef PONOS_STRUCTURES_TETRAHEDRON_H
#define PONOS_STRUCTURES_TETRAHEDRON_H

#include <ponos/structures/raw_mesh.h>
#include <vector>
#include <algorithm>
#include <map>
#include <set>

namespace ponos {

/** Represents a mesh of tetrahedra with topological information for fast
 * neighbourhood queries.
 * \tparam V Vertex data
 * \tparam F Face data
 * \tparam E Edge data
 * \tparam P Tetrahedron data
 */
template<typename V = float, typename F = float, typename E = float,
    typename P = float>
class TMesh {
public:
  struct Vertex {
    Vertex(float x, float y, float z) : position(Point3(x, y, z)) {}
    explicit Vertex(const Point3 &p) : position(p) {}
    Point3 position; //!< vertex position
    Normal normal;   //!< vertex normal
    std::set<size_t> edges; //!< edges connected to this vertex
    V data;          //!< vertex user's data
  };
  struct Edge {
    explicit Edge(size_t _a, size_t _b) : a(_a), b(_b), face(-1) {}
    size_t a;    //!< vertex a -> smaller index
    size_t b;    //!< vertex b -> greater index
    int face; //!< a face that is connected to it
    E data;   //!< edge user's data
  };
  struct Face {
    Face() : hface(-1), face3(-1) {
      for (size_t i = 0; i < 3; i++)
        edges[i] = 0, neighbours[i] = -1;
    }
    int hface;         //!< half-face neighbour
    int face3;         //!< index to tetrahedron
    size_t edges[3];   //!< index to face's edges list (starting by the smallest
                       //!< vertex index)
    int neighbours[3]; //!< neighbours
    F data;            //!< face user's data
    Normal normal;     //!< face's normal
  };
  struct Face3 {
    explicit Face3(size_t _face) : face(_face) {}
    size_t face; //!< one half face
    P data;   //!< face3 users'data
  };
  /// \param rm input mesh
  explicit TMesh(const RawMesh &rm);
  /// \param t tetrahedron index
  /// \return list of 4 vertex indices
  std::vector<size_t> tetrahedronVertices(size_t t) const;
  /// \param t tetrahedron index
  /// \return list of 4 face indices
  std::vector<size_t> tetrahedronFaces(size_t t) const;
  /// \param t tetrahedron index
  /// \return list of 3 vertex indices
  std::vector<size_t> faceVertices(size_t f) const;
  /// \param t tetrahedron index
  /// \return list of existent tetrahedra indices
  std::vector<size_t> tetrahedronNeighbours(size_t t) const;
  /// \param v vertex index
  /// \return list of neighbour vertices
  std::vector<size_t> vertexNeighbours(size_t v) const;
  /// \param v vertex index
  /// \return list of neighbour tetrahedra
  std::vector<size_t> vertexTetrahedra(size_t v) const;
  std::vector<Vertex> vertices;
  std::vector<Edge> edges;
  std::vector<Face> faces;
  std::vector<Face3> tetrahedron;
private:
  struct IndexContainer {
    std::vector<int> data;
  };
  IndexContainer edgesFaces_;
};

#include <ponos/structures/tetrahedron_mesh.inl>

} // ponos namespace

#endif // PONOS_STRUCTURES_HALF_EDGE_3_H
