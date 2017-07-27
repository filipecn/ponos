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

#ifndef PONOS_STRUCTURES_HALF_EDGE_H
#define PONOS_STRUCTURES_HALF_EDGE_H

#include "geometry/point.h"
#include "structures/raw_mesh.h"

#include <functional>
#include <map>

namespace ponos {

template <class T, int D, typename V = float, typename F = float,
          typename E = float>
class HEMesh {
public:
  HEMesh() {}
  HEMesh(const RawMesh *rm) {
    ASSERT_FATAL(rm->meshDescriptor.elementSize == 3);
    for (size_t i = 0; i < rm->vertexDescriptor.count; i++)
      addVertex(Point<float, 2>(
          {rm->vertices[i * rm->vertexDescriptor.elementSize + 0],
           rm->vertices[i * rm->vertexDescriptor.elementSize + 1]}));
    std::map<std::pair<int, int>, int> m;
    for (size_t i = 0; i < rm->meshDescriptor.count; i++) {
      int faceEdges[3];
      for (size_t e = 0; e < 3; e++) {
        std::pair<int, int> edge = std::make_pair(
            rm->indices[i * rm->meshDescriptor.elementSize + e].vertexIndex,
            rm->indices[i * rm->meshDescriptor.elementSize + (e + 1) % 3]
                .vertexIndex);
        std::pair<int, int> edgePair = std::make_pair(edge.second, edge.first);
        auto itA = m.find(edge);
        auto itB = m.find(edgePair);
        ASSERT_FATAL((itA == m.end() && itB == m.end()) ||
                     (itA != m.end() && itB != m.end()));
        if (itA == m.end()) {
          int eA = addEdge(edge.first, edge.second);
          m[edge] = eA;
          m[edgePair] = eA + 1;
        }
        faceEdges[e] = m[edge];
      }
      addFace({faceEdges[0], faceEdges[1], faceEdges[2]});
    }
  }
  struct Vertex {
    Vertex(T x, T y) : position(Point<T, D>({x, y})), edge(-1) {}
    Vertex(const Point<T, D> &p) : position(p), edge(-1) {}
    Point<T, D> position; //!< HE vertex position
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
    int edge; //!< one half edge
    F data;   //!< face user's data
  };
  void setVertexData(size_t v, const V &data) { vertices[v].data = data; }
  const std::vector<Vertex> &getVertices() const { return vertices; }
  const std::vector<Edge> &getEdges() const { return edges; }
  const std::vector<Face> &getFaces() const { return faces; }
  size_t addVertex(const Point<T, D> &p) {
    vertices.emplace_back(p);
    return vertices.size() - 1;
  }
  size_t addEdge(int a, int b) {
    vec2 v = vertices[b].position.floatXY() - vertices[a].position.floatXY();
    int e1 = edges.size();
    int e2 = e1 + 1;
    edges.resize(edges.size() + 2);

    // std::cout << "for edge " << e1 << std::endl;
    // std::cout << "vertices " << a << " and " << b << std::endl;
    edges[e1].pair = e2;
    edges[e1].face = -1;
    edges[e1].orig = a;
    edges[e1].dest = b;
    edges[e1].next = findNextEdge(-v, b, true);
    edges[e1].prev = findNextEdge(v, a, false);
    // std::cout << "next " << edges[e1].next << std::endl;
    // std::cout << "prev " << edges[e1].prev << std::endl;
    if (edges[e1].next < 0)
      edges[e1].next = e2;
    if (edges[e1].prev < 0)
      edges[e1].prev = e2;
    // std::cout << "next " << edges[e1].next << std::endl;
    // std::cout << "prev " << edges[e1].prev << std::endl;

    // std::cout << "for edge " << e2 << std::endl;
    edges[e2].pair = e1;
    edges[e2].face = -1;
    edges[e2].orig = b;
    edges[e2].dest = a;
    edges[e2].next = findNextEdge(v, a, true);
    edges[e2].prev = findNextEdge(-v, b, false);
    // std::cout << "next " << edges[e2].next << std::endl;
    // std::cout << "prev " << edges[e2].prev << std::endl;
    if (edges[e2].next < 0)
      edges[e2].next = e1;
    if (edges[e2].prev < 0)
      edges[e2].prev = e1;
    // std::cout << "next " << edges[e2].next << std::endl;
    // std::cout << "prev " << edges[e2].prev << std::endl;

    if (edges[e1].next >= 0)
      edges[edges[e1].next].prev = e1;
    if (edges[e1].prev >= 0)
      edges[edges[e1].prev].next = e1;

    if (edges[e2].next >= 0)
      edges[edges[e2].next].prev = e2;
    if (edges[e2].prev >= 0)
      edges[edges[e2].prev].next = e2;

    if (vertices[a].edge < 0)
      vertices[a].edge = e1;
    if (vertices[b].edge < 0)
      vertices[b].edge = e2;
    // std::cout << "vertex " << a << " has edge " << vertices[a].edge <<
    // std::endl;
    // std::cout << "vertex " << b << " has edge " << vertices[b].edge <<
    // std::endl;
    // std::cout << "Edge created\n";
    // std::cout << "edge " << e1 << std::endl;
    // std::cout << "a " << edges[e1].orig << " b "
    // 	<< edges[e1].dest << std::endl;
    // std::cout << "next " << edges[e1].next << " ";
    // std::cout << "prev " << edges[e1].prev << " ";
    // std::cout << "pair " << edges[e1].pair << " ";
    // std::cout << "face " << edges[e1].face << "\n";
    // std::cout << std::endl;
    // std::cout << "edge " << e2 << std::endl;
    // std::cout << "a " << edges[e2].orig << " b "
    // 	<< edges[e2].dest << std::endl;
    // std::cout << "next " << edges[e2].next << " ";
    // std::cout << "prev " << edges[e2].prev << " ";
    // std::cout << "pair " << edges[e2].pair << " ";
    // std::cout << "face " << edges[e2].face << "\n";
    // std::cout << std::endl;
    return e1;
  }
  size_t addFace(std::initializer_list<int> e) {
    Face f;
    f.edge = *e.begin();
    for (auto edge = e.begin(); edge != e.end(); edge++)
      edges[*edge].face = faces.size();
    faces.emplace_back(f);
    return faces.size() - 1;
  }
  void traverseFacesFromVertex(int v, std::function<void(int f)> f) {
    traverseEdgesFromVertex(v, [this, f](int e) {
      if (edges[e].face >= 0)
        f(edges[e].face);
    });
  }
  void traverseEdgesFromVertex(int v, std::function<void(int e)> f) {
    if (vertices[v].edge >= 0) {
      int begin = vertices[v].edge;
      int he = begin;
      do {
        f(he);
        int ee = vertexNext(he);
        if (ee == he)
          break;
        he = ee;
      } while (he != begin);
    }
  }
  void traverseEdgesToVertex(int v, std::function<void(int e)> f) {
    if (vertices[v].edge >= 0) {
      int begin = vertices[v].edge;
      int he = begin;
      do {
        f(edges[he].pair);
        int ee = vertexNext(he);
        if (ee == he)
          break;
        he = ee;
      } while (he != begin);
    }
  }
  void traversePolygonEdges(int f, std::function<void(int e)> g) const {
    if (faces[f].edge >= 0) {
      int begin = faces[f].edge;
      int he = begin;
      do {
        g(he);
        he = edges[he].next;
      } while (he != begin);
    }
  }
  size_t vertexCount() const { return vertices.size(); }
  size_t faceCount() const { return faces.size(); }

private:
  /** \brief
   * \param v (always leaving vertex)
   * \param vertex
   * \param incident the original v is incident to vertex?
   * \returns
   */
  int findNextEdge(vec2 v, int vertex, bool incident) {
    vec2 nv = normalize(v);
    double xAngle = -(1 << 20);
    double vAngle = ponos::atanPI_2(v.y, v.x);
    int xAngleId = -1;
    int nextEdge = -1;
    if (incident) {
      xAngle = -(1 << 20);
      double nextAngle = -(1 << 20);
      traverseEdgesFromVertex(vertex, [nv, &xAngle, &xAngleId, &vAngle,
                                       &nextAngle, &nextEdge, this](int e) {
        vec2 curVec = normalize(vertices[edges[e].dest].position.floatXY() -
                                vertices[edges[e].orig].position.floatXY());
        double angle = ponos::atanPI_2(curVec.y, curVec.x);
        if (angle > xAngle)
          xAngle = angle, xAngleId = e;
        if (angle < vAngle && angle > nextAngle)
          nextAngle = angle, nextEdge = e;
      });
    } else {
      xAngle = (1 << 20);
      double nextAngle = (1 << 20);
      traverseEdgesToVertex(vertex, [nv, &xAngle, &xAngleId, &vAngle,
                                     &nextAngle, &nextEdge, this](int e) {
        vec2 curVec = normalize(vertices[edges[e].orig].position.floatXY() -
                                vertices[edges[e].dest].position.floatXY());
        double angle = ponos::atanPI_2(curVec.y, curVec.x);
        // std::cout << e << " " << TO_DEGREES(angle) << "\n";
        if (angle < xAngle)
          xAngle = angle, xAngleId = e;
        if (angle > vAngle && angle < nextAngle)
          nextAngle = angle, nextEdge = e;
      });
    }
    if (nextEdge < 0)
      nextEdge = xAngleId;
    return nextEdge;
  }
  int vertexNext(int e) { return edges[edges[e].pair].next; }
  int vertexPrev(int e) { return edges[edges[e].prev].pair; }

  std::vector<Vertex> vertices;
  std::vector<Edge> edges;
  std::vector<Face> faces;
};

typedef HEMesh<float, 2> HEMesh2DF;

} // ponos namespace

#endif // PONOS_STRUCTURES_HALF_EDGE_H
