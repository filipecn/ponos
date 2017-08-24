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

#ifndef PONOS_STRUCTURES_HALF_EDGE_3_H
#define PONOS_STRUCTURES_HALF_EDGE_3_H

#include "structures/half_edge.h"

namespace ponos {

/** Represents a 3D mesh of polygons with topological information for fast
 * neighbourhood queries.
 * \tparam T Coordinate type
 * \tparam D Vertex dimension
 * \tparam V Vertex data
 * \tparam F Face data
 * \tparam E Edge data
 * \tparam P Tetrahedron data
 */
template <class T = float, size_t D = 3, typename V = float, typename F = float,
          typename E = float, typename P = float>
class HEMesh3 {
public:
  struct Vertex {
    Vertex(T x, T y) : position(Point<T, D>({x, y})), edge(-1) {}
    Vertex(const Point<T, D> &p) : position(p), edge(-1) {}
    Point<T, D> position; //!< HE vertex position
    Vector<T, D> normal;  //!< HE vertex normal
    int edge;             //!< HE connected to this vertex
    V data;               //!< vertex user's data
  };
  struct Edge {
    Edge() { orig = dest = pair = face = next = prev = -1; }
    int orig; //!< origin
    int dest; //!< destination
    int pair; //!< oposite half edge
    int face; //!< face on the left
    int next; //!< successor HE
    int prev; //!< predecessor HE
    E data;   //!< edge user's data
  };
  struct Face {
    Face() : edge(-1) {}
    int edge;            //!< one half edge
    F data;              //!< face user's data
    Vector<T, D> normal; //!< face's normal
  };
};

} // ponos namespace

#endif // PONOS_STRUCTURES_HALF_EDGE_3_H
