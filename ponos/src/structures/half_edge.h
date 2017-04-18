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

template <class T, int D> class HEMesh {
public:
  HEMesh() {}
  HEMesh(const RawMesh *rm) {
    ASSERT_FATAL(rm->vertexDescriptor.elementSize == 2);
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
      std::cout << "adding face:\n";
      std::cout << faceEdges[0] << " ";
      std::cout << faceEdges[1] << " ";
      std::cout << faceEdges[2] << "\n";
      addFace({faceEdges[0], faceEdges[1], faceEdges[2]});
    }
  }

  struct Vertex {
    Vertex(const Point<T, D> &p) : position(p), edge(-1) {}
    Point<T, D> position;
    int edge; //!< HE connected to this vertex
  };
  struct Edge {
    Edge() { orig = dest = pair = face = next = prev = -1; }
    int orig; //!< origin
    int dest; //!< origin
    int pair; //!< oposite half edge
    int face; //!< face on the left
    int next; //!< successor HE
    int prev; //!< predecessor HE
  };
  struct Face {
    int edge; //!< one half edge
  };
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

    std::cout << "for edge " << e1 << std::endl;
    edges[e1].pair = e2;
    edges[e1].face = -1;
    edges[e1].orig = a;
    edges[e1].dest = b;
    edges[e1].next = findNextEdge(-v, b, false);
    edges[e1].prev = findNextEdge(v, a, true);
    std::cout << "next " << edges[e1].next << std::endl;
    std::cout << "prev " << edges[e1].prev << std::endl;
    if (edges[e1].next < 0)
      edges[e1].next = e2;
    if (edges[e1].prev < 0)
      edges[e1].prev = e2;
    std::cout << "next " << edges[e1].next << std::endl;
    std::cout << "prev " << edges[e1].prev << std::endl;

    std::cout << "for edge " << e2 << std::endl;
    edges[e2].pair = e1;
    edges[e2].face = -1;
    edges[e2].orig = b;
    edges[e2].dest = a;
    edges[e2].next = findNextEdge(-v, a, false);
    edges[e2].prev = findNextEdge(v, b, true);
    std::cout << "next " << edges[e2].next << std::endl;
    std::cout << "prev " << edges[e2].prev << std::endl;
    if (edges[e2].next < 0)
      edges[e2].next = e1;
    if (edges[e2].prev < 0)
      edges[e2].prev = e1;
    std::cout << "next " << edges[e2].next << std::endl;
    std::cout << "prev " << edges[e2].prev << std::endl;

    if (edges[e1].next >= 0)
      edges[edges[e1].next].prev = e1;
    if (edges[e1].prev >= 0)
      edges[edges[e1].prev].next = e1;

    if (edges[e1].next >= 0)
      std::cout << "conected next-previous " << edges[e1].next << " "
                << edges[edges[e1].next].prev << std::endl;
    if (edges[e1].prev >= 0)
      std::cout << "conected previous-next " << edges[e1].prev << " "
                << edges[edges[e1].prev].next << std::endl;

    if (edges[e2].next >= 0)
      edges[edges[e2].next].prev = e2;
    if (edges[e2].prev >= 0)
      edges[edges[e2].prev].next = e2;

    if (edges[e2].next >= 0)
      std::cout << "conected next-previous " << edges[e2].next << " "
                << edges[edges[e2].next].prev << std::endl;
    if (edges[e2].prev >= 0)
      std::cout << "conected previous-next " << edges[e2].prev << " "
                << edges[edges[e2].prev].next << std::endl;

    if (vertices[a].edge < 0)
      vertices[a].edge = e1;
    if (vertices[b].edge < 0)
      vertices[b].edge = e2;

    std::cout << "Edge created\n";
    std::cout << edges[e1].next << " ";
    std::cout << edges[e1].prev << " ";
    std::cout << edges[e1].pair << " ";
    std::cout << edges[e1].face << "\n";
    std::cout << std::endl;
    std::cout << edges[e2].next << " ";
    std::cout << edges[e2].prev << " ";
    std::cout << edges[e2].pair << " ";
    std::cout << edges[e2].face << "\n";
    std::cout << std::endl;
    return e1;
  }
  size_t addFace(std::initializer_list<int> e) {
    Face f;
    // int lastEdge = -1;
    // for (auto it = e.begin(); it != e.end(); ++it) {
    //  edges[*it].face = faces.size();
    //  if (lastEdge >= 0)
    //    edges[lastEdge].next = *it;
    //  edges[*it].prev = lastEdge;
    //  lastEdge = *it;
    //}
    // edges[*e.begin()].prev = lastEdge;
    // edges[lastEdge].next = *e.begin();
    // f.edge = lastEdge;
    f.edge = *e.begin();
    faces.emplace_back(f);
    return faces.size() - 1;
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
  void traversePolygonEdges(int f, std::function<void(int e)> g) {
    if (faces[f].edge >= 0) {
      int begin = faces[f].edge;
      int he = begin;
      do {
        g(he);
        he = edges[he].next;
      } while (he != begin);
    }
  }

private:
  /** \brief
   * \param v (always leaving v)
   * \param vertex
   * \param incident (search for a incident edge?)
   * \returns
   */
  int findNextEdge(vec2 v, int vertex, bool incident) {
    vec2 nv = normalize(v);
    float maxAngle = -(1 << 20);
    int nextEdge = -1;
    std::cout << "find next " << incident << std::endl;
    if (incident)
      traverseEdgesToVertex(vertex, [nv, &maxAngle, &nextEdge, this](int e) {
        vec2 curVec = normalize(vertices[edges[e].dest].position.floatXY() -
                                vertices[edges[e].orig].position.floatXY());
        float angle = cross(nv, curVec);
        std::cout << e << " ";
        if (angle > maxAngle) {
          maxAngle = angle;
          nextEdge = e;
        }
      });
    else
      traverseEdgesFromVertex(vertex, [nv, &maxAngle, &nextEdge, this](int e) {
        vec2 curVec = normalize(vertices[edges[e].dest].position.floatXY() -
                                vertices[edges[e].orig].position.floatXY());
        float angle = cross(-nv, curVec);
        std::cout << e << " ";
        if (angle > maxAngle) {
          maxAngle = angle;
          nextEdge = e;
        }
      });
    std::cout << " found edge " << nextEdge << std::endl;
    return nextEdge;
  }
  int vertexNext(int e) { return edges[edges[e].pair].next; }
  int vertexPrev(int e) { return edges[edges[e].prev].pair; }

  std::vector<Vertex> vertices;
  std::vector<Edge> edges;
  std::vector<Face> faces;
};

} // ponos namespace

#endif // PONOS_STRUCTURES_HALF_EDGE_H
