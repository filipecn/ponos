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

#include "structures/raw_mesh.h"

namespace ponos {

/** Represents a mesh of tetrahedra with topological information for fast
 * neighbourhood queries.
 * \tparam V Vertex data
 * \tparam F Face data
 * \tparam E Edge data
 * \tparam P Tetrahedron data
 */
template <typename V = float, typename F = float, typename E = float,
          typename P = float>
class TMesh {
public:
  struct Vertex {
    Vertex(float x, float y, float z) : position(Point3(x, y, z)), edge(-1) {}
    Vertex(const Point3 &p) : position(p), edge(-1) {}
    Point3 position; //!< vertex position
    Normal normal;   //!< vertex normal
    int edge;        //!< edge connected to this vertex
    V data;          //!< vertex user's data
  };
  struct Edge {
    Edge() { a = b = face = -1; }
    int a;    //!< vertex a
    int b;    //!< vertex b
    int face; //!< face on the left
    E data;   //!< edge user's data
  };
  struct Face {
    Face() : face3(-1) {}
    int face3;         //!< index to tetrahedron
    int edges[3];      //!< index to face's edges list
    int neighbours[3]; //!< neighbours
    F data;            //!< face user's data
    Normal normal;     //!< face's normal
  };
  struct Face3 {
    Face3() : face(-1) {}
    int face; //!< one half face
    P data;   //!< face3 users'data
  };

  TMesh(const RawMesh &rm);

private:
  struct IndexContainer {
    std::vector<int> data;
  };
  IndexContainer edgesFaces;
  std::vector<Vertex> vertices;
};

#include "structures/tetrahedron_mesh.inl"

} // ponos namespace

#endif // PONOS_STRUCTURES_HALF_EDGE_3_H
