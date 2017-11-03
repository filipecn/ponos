#ifndef PONOS_STRUCTURES_BREP_H
#define PONOS_STRUCTURES_BREP_H

#include <ponos/common/defs.h>
#include <ponos/geometry/point.h>

#include <vector>

namespace ponos {
/*
l_s     r_p
   \   /
    \ /
    x b
     |
  l  |  r
     |
    x a
    / \
   /   \
l_p     r_s
*/
template <class T, int D> class Brep {
public:
  struct Vertex {
    Point<T, D> pos;
    std::vector<int> edges;
    Vertex() {}
    Vertex(Point<T, D> p) { pos = p; }
  };
  struct Edge {
    // vertices
    int a, b;
    // faces
    int l, r;
    // left traverse
    int leftPred, leftSucc;
    // right traverse
    int rightPred, rightSucc;
    Edge() { a = b = l = r = leftPred = leftSucc = rightPred = rightSucc = -1; }
    Edge(uint _a, uint _b) {
      a = _a;
      b = _b;
      l = r = leftPred = leftSucc = rightPred = rightSucc = -1;
    }
  };

  int addVertex(Vertex v);
  int addVertex(Point<T, D> p);
  void removeVertex(uint i);
  int addEdge(Edge e);
  int addEdge(uint a, uint b);
  void removeEdge(uint i);
  int addFace(const std::vector<int> &vs);
  void removeFace(uint i);

  std::vector<Vertex> vertices;
  std::vector<int> faces;
  std::vector<Edge> edges;

private:
  // update vertex edges information
  void updateVertex(uint v);
};

template <class T, int D> int Brep<T, D>::addVertex(Vertex v) {
  vertices.emplace_back(v);
  return static_cast<int>(vertices.size()) - 1;
}
template <class T, int D> int Brep<T, D>::addVertex(Point<T, D> p) {
  vertices.emplace_back(p);
  return static_cast<int>(vertices.size()) - 1;
}

template <class T, int D> void Brep<T, D>::removeVertex(uint i) {}

template <class T, int D> int Brep<T, D>::addEdge(Edge e) {
  edges.emplace_back(e);
  return static_cast<int>(edges.size()) - 1;
}

template <class T, int D> int Brep<T, D>::addEdge(uint a, uint b) {
  edges.emplace_back(a, b);
  updateVertex(a);
  updateVertex(b);
  return static_cast<int>(edges.size()) - 1;
}

template <class T, int D> void Brep<T, D>::removeEdge(uint i) {}

template <class T, int D> int Brep<T, D>::addFace(const std::vector<int> &vs) {
  UNUSED_VARIABLE(vs);
  /*int newFace = faces.size();
  bool manifold = true;
  for(int v : vs) {
          if(edges[vertices[v].edge].l >= 0) {
                  manifold = false;
                  break;
          }
  }
  if(!manifold)
          return -1;
  for(int v : vs)
          edges[vertices[v].edge].l = newFace;
  faces.emplace_back(vs[0]);*/
  return 0;
}

template <class T, int D> void Brep<T, D>::removeFace(uint i) {
  UNUSED_VARIABLE(i);
}

template <class T, int D> void Brep<T, D>::updateVertex(uint v) {
  UNUSED_VARIABLE(v);
}
} // ponos namespace

#endif
