#ifndef PONOS_FAST_SWEEP_H
#define PONOS_FAST_SWEEP_H

#include <queue>

#include <ponos/geometry/queries.h>
#include <ponos/geometry/vector.h>
#include <ponos/structures/half_edge.h>
#include <ponos/structures/tetrahedron_mesh.h>

namespace ponos {

inline void solveDistance(float p, float q, float &r) {
  float d = std::min(p, q) + 1;
  if (d > std::max(p, q))
    d = (p + q + sqrtf(2 - SQR(p - q))) / 2;
  if (d < r)
    r = d;
}

template <typename GridType, typename MaskType, typename T>
void fastSweep2D(GridType *grid, GridType *distances, MaskType *mask,
                 T MASK_VALUE) {
  // +1 +1
  for (int j = 1; j < grid->getDimensions()[1]; j++)
    for (int i = 1; i < grid->getDimensions()[0]; i++) {
      if ((*mask)(i, j) != MASK_VALUE)
        solveDistance((*grid)(i - 1, j), (*grid)(i, j - 1), (*grid)(i, j));
    }
  // -1 -1
  for (int j = grid->getDimensions()[1] - 2; j >= 0; j--)
    for (int i = grid->getDimensions()[0] - 2; i >= 0; i--) {
      if ((*mask)(i, j) != MASK_VALUE)
        solveDistance((*grid)(i + 1, j), (*grid)(i, j + 1), (*grid)(i, j));
    }
  // +1 -1
  for (int j = grid->getDimensions()[1] - 2; j >= 0; j--)
    for (int i = 1; i < grid->getDimensions()[0]; i++) {
      if ((*mask)(i, j) != MASK_VALUE)
        solveDistance((*grid)(i - 1, j), (*grid)(i, j + 1), (*grid)(i, j));
    }
  // -1 +1
  for (int j = 1; j < grid->getDimensions()[1]; j++)
    for (int i = grid->getDimensions()[0] - 2; i >= 0; i--) {
      if ((*mask)(i, j) != MASK_VALUE)
        solveDistance((*grid)(i + 1, j), (*grid)(i, j - 1), (*grid)(i, j));
    }
}

template <typename GridType, typename MaskType, typename T>
void sweep_y(GridType *grid, GridType *phi, MaskType *mask, T MASK_VALUE,
             int i0, int i1, int j0, int j1) {
  int di = (i0 < i1) ? 1 : -1, dj = (j0 < j1) ? 1 : -1;
  float dp, dq, alpha;
  for (int j = j0; j != j1; j += dj)
    for (int i = i0; i != i1; i += di)
      if ((*mask)(i, j - 1) == MASK_VALUE && (*mask)(i, j) == MASK_VALUE) {
        dq = dj * ((*phi)(i, j) - (*phi)(i, j - 1));
        if (dq < 0)
          continue; // not useful on this sweep direction
        dp = 0.5 * ((*phi)(i, j - 1) + (*phi)(i, j) - (*phi)(i - di, j - 1) -
                    (*phi)(i - di, j));
        if (dp < 0)
          continue; // not useful on this sweep direction
        if (dp + dq == 0)
          alpha = 0.5;
        else
          alpha = dp / (dp + dq);
        (*grid)(i, j) =
            alpha * (*grid)(i - di, j) + (1 - alpha) * (*grid)(i, j - dj);
      }
}

template <typename GridType, typename MaskType, typename T>
void sweep_x(GridType *grid, GridType *phi, MaskType *mask, T MASK_VALUE,
             int i0, int i1, int j0, int j1) {
  int di = (i0 < i1) ? 1 : -1, dj = (j0 < j1) ? 1 : -1;
  float dp, dq, alpha;
  for (int j = j0; j != j1; j += dj)
    for (int i = i0; i != i1; i += di)
      if ((*mask)(i - 1, j) == MASK_VALUE && (*mask)(i, j) == MASK_VALUE) {
        dq = dj * ((*phi)(i, j) - (*phi)(i - 1, j));
        if (dq < 0)
          continue; // not useful on this sweep direction
        dp = 0.5 * ((*phi)(i - 1, j) + (*phi)(i, j) - (*phi)(i - 1, j - dj) -
                    (*phi)(i, j - dj));
        if (dp < 0)
          continue; // not useful on this sweep direction
        if (dp + dq == 0)
          alpha = 0.5;
        else
          alpha = dp / (dp + dq);
        (*grid)(i, j) =
            alpha * (*grid)(i - di, j) + (1 - alpha) * (*grid)(i, j - dj);
      }
}

template <typename GridType, typename MaskType, typename T>
void fastMarch2D(GridType *phi, MaskType *mask, T MASK_VALUE,
                 std::vector<index2> points) {
  UNUSED_VARIABLE(MASK_VALUE);
  RegularGrid2D<char> frozen((*phi).getDimensions(), 0);
  frozen.setAll(0);
  struct Point_ {
    Point_(int I, int J, float D) : i(I), j(J), t(D) {}
    int i, j;
    float t;
  };
  auto cmp = [](Point_ a, Point_ b) { return a.t > b.t; };
  std::priority_queue<Point_, std::vector<Point_>, decltype(cmp)> q(cmp);
  (*phi).setAll(1 << 16);
  for (auto p : points) {
    (*phi)(p) = 0;
    q.push(Point_(p[0], p[1], (*phi)(p)));
  }
  while (!q.empty()) {
    Point_ p = q.top();
    q.pop();
    frozen(p.i, p.j) = 2;
    (*phi)(p.i, p.j) = p.t;
    index2 dir[4] = {index2(-1, 0), index2(1, 0), index2(0, -1), index2(0, 1)};
    for (int i = 0; i < 4; i++) {
      index2 ij = index2(p.i, p.j) + dir[i];
      if (ij[0] < 0 || ij[0] >= frozen.getDimensions()[0] || ij[1] < 0 ||
          ij[1] >= frozen.getDimensions()[1] || frozen(ij) || !(*mask)(ij))
        continue;
      float T1 = std::min((*phi)(ij + dir[0]), (*phi)(ij + dir[1]));
      float T2 = std::min((*phi)(ij + dir[2]), (*phi)(ij + dir[3]));
      float Tmin = std::min(T1, T2);
      float d = fabs(T1 - T2);
      if (d < 1.f)
        (*phi)(ij) = (T1 + T2 + sqrtf(2.f * SQR(1.f) - SQR(d))) / 2.f;
      else
        (*phi)(ij) = Tmin + 1.f;
      q.push(Point_(ij[0], ij[1], (*phi)(ij)));
      frozen(ij) = 1;
    }
  }
}

inline void fastMarch2D(HEMesh2DF *phi, std::vector<size_t> points) {
  const std::vector<HEMesh2DF::Edge> &edges = phi->getEdges();
  const std::vector<HEMesh2DF::Vertex> &vertices = phi->getVertices();
  size_t vc = phi->vertexCount();
  std::vector<char> frozen(vc, 0);
  std::vector<int> closestPoint(vc, -1);
  struct Point_ {
    Point_(size_t _v, float D) : v(_v), t(D) {}
    size_t v;
    float t;
  };
  auto cmp = [](Point_ a, Point_ b) { return a.t > b.t; };
  std::priority_queue<Point_, std::vector<Point_>, decltype(cmp)> q(cmp);
  for (size_t i = 0; i < vc; i++)
    phi->setVertexData(i, 1 << 16);
  for (auto p : points) {
    phi->setVertexData(p, 0.f);
    q.push(Point_(p, 0.f));
    closestPoint[p] = p;
  }
  while (!q.empty()) {
    Point_ p = q.top();
    q.pop();
    frozen[p.v] = 2;
    phi->setVertexData(p.v, p.t);
    // iterate all neighbours
    phi->traverseEdgesFromVertex(p.v, [&q, &phi, &edges, &vertices, &frozen,
                                       &closestPoint](int e) {
      if (frozen[edges[e].dest])
        return;
      phi->setVertexData(edges[e].dest, 1 << 16);
      // iterate neighbours from neighbours
      phi->traverseEdgesFromVertex(
          edges[e].dest,
          [&phi, &edges, &vertices, &frozen, &closestPoint](int e2) {
            int a = edges[e2].orig;
            // test against vertex
            int b = edges[e2].dest;
            float bDist = distance(vertices[a].position, vertices[b].position);
            if (vertices[b].data + bDist < vertices[a].data) {
              phi->setVertexData(a, vertices[b].data + bDist);
              closestPoint[a] = b;
            }
            // test against oposite edge
            int c = edges[edges[e2].next].dest;
            if (closestPoint[b] >= 0 && closestPoint[b] != b) {
              Segment2 s(vertices[b].position, vertices[c].position);
              Ray2 r(vertices[a].position,
                     vertices[closestPoint[b]].position - vertices[a].position);
              float dist = distance(vertices[a].position,
                                    vertices[closestPoint[b]].position);
              if (ray_segment_intersection(r, s) &&
                  dist + vertices[closestPoint[b]].data < vertices[a].data) {
                phi->setVertexData(a, vertices[closestPoint[b]].data + dist);
                closestPoint[a] = closestPoint[b];
              }
            }
            if (closestPoint[c] >= 0 && closestPoint[c] != c) {
              Segment2 s(vertices[b].position, vertices[c].position);
              Ray2 r(vertices[a].position,
                     vertices[closestPoint[c]].position - vertices[a].position);
              float dist = distance(vertices[a].position,
                                    vertices[closestPoint[c]].position);
              if (ray_segment_intersection(r, s) &&
                  dist + vertices[closestPoint[c]].data < vertices[a].data) {
                phi->setVertexData(a, vertices[closestPoint[c]].data + dist);
                closestPoint[a] = closestPoint[c];
              }
            }
          });
      q.push(Point_(edges[e].dest, vertices[edges[e].dest].data));
      frozen[edges[e].dest] = 1;
    });
  }
}

/// Propagates distances from sources to vertices.
/// \tparam MeshType tetrahedra mesh type
/// \param input input mesh pointer
/// \param sources source vertices indices
/// \param out output mesh pointer
template <typename MeshType = TMesh<>>
void fastMarchTetraheda(const MeshType *in, std::vector<size_t> sources,
                        MeshType *out) {
  // uses an phantom closest point to all (it will always have infinity distance
  std::vector<size_t> closest(out->vertices.size() + 1, out->vertices.size());
  std::vector<double> distances(out->vertices.size() + 1, INFINITY);
  // in case source is not directly visible, the phanton index is used
  /*auto computeClosestPoint = [&](size_t p, size_t s, double &d) -> size_t {
    std::cerr << "is visible " << p << " " << s << std::endl;
    ray3 r(in->vertices[s].position, normalize(in->vertices[p].position -
        in->vertices[s].position));
    d = INFINITY;
    // iterate all neighboring tetrahedra
    auto ts = in->vertexTetrahedra(p);
    // r must intersect at least one of its interior faces
    for (auto t : ts) {
      auto fs = in->tetrahedronFaces(t);
      for (auto f : fs)
        if (in->faces[f].hface >= 0) {
          auto v = in->faceVertices(f);
          if (triangle_ray_intersection(in->vertices[v[0]].position,
                                        in->vertices[v[1]].position,
                                        in->vertices[v[2]].position,
                                        r)) {
            d = distance(in->vertices[s].position, in->vertices[p].position);
            return s;
          } else {
            auto cp = closest_point_triangle(
                in->vertices[s].position, in->vertices[v[0]].position,
                in->vertices[v[1]].position, in->vertices[v[2]].position);
            std::cerr << "cp " << cp << " on " << v[0] << " " << v[1] << " " <<
  v[2] << std::endl; std::cerr << "distances:\n "; std::cerr <<
  distance(in->vertices[s].position, cp) << " " << distance(cp,
  in->vertices[p].position) << std::endl; double dist =
  distance(in->vertices[s].position, cp) + distance(cp,
  in->vertices[p].position); d = std::min(d, dist);
          }
        }
    }
    std::cerr << "not visible\n";
    return in->vertices.size();
  };*/
  std::queue<size_t> q;
  // initialize queue with sources
  for (auto s : sources) {
    distances[s] = 0.f;
    closest[s] = s;
    q.push(s);
  }
  while (!q.empty()) {
    size_t cur = q.front();
    std::cerr << "CUR " << cur << std::endl;
    q.pop();
    auto neighbours = in->vertexNeighbours(cur);
    std::cerr << "with neighbors\n";
    for (auto n : neighbours)
      std::cerr << n << " ";
    std::cerr << std::endl;
    // for each neighbour, try to update cur vertex by getting their closest
    // source and considering the source position and mesh
    for (auto n : neighbours) {
      auto d = distance(in->vertices[n].position, in->vertices[cur].position) +
               distances[cur];
      if (d < distances[n]) {
        distances[n] = d;
        q.push(n);
      }
      /*if (closest[cur] == cur) {
        // we are in a source node, so all neighbors have direct visibility
        auto d = distance(in->vertices[n].position,
      in->vertices[closest[cur]].position); if (d < distances[n]) { distances[n]
      = d; closest[n] = cur; q.push(n);
        }
        continue;
      }
      if (closest[cur] == out->vertices.size() || closest[cur] == closest[n])
        continue;
      double d = INFINITY;
      size_t cp = computeClosestPoint(n, closest[cur], d);
      std::cerr << "distance " << d << " -> " << cp << std::endl;
      if (d < distances[n]) {
        distances[n] = d;
        closest[n] = cp;
        q.push(n);
      }*/
    }
  }
  for (size_t i = 0; i < out->vertices.size(); i++)
    out->vertices[i].data = distances[i];
}

} // namespace ponos

#endif // PONOS_FAST_SWEEP_H
