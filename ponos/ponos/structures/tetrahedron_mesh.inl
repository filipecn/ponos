/*
 * Copyright (c) 2018 FilipeCN
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

template<typename V, typename F, typename E, typename P>
TMesh<V, F, E, P>::TMesh(const RawMesh &rm) {
#ifdef TETGEN_INCLUDED
  if (rm.primitiveType == GeometricPrimitiveType::TRIANGLES) {
    return;
  }
#endif
  FATAL_ASSERT(rm.meshDescriptor.elementSize == 4);
  FATAL_ASSERT(rm.positionDescriptor.elementSize == 3);
  FATAL_ASSERT(rm.primitiveType == GeometricPrimitiveType::TETRAHEDRA);
  // add vertices on the same order as found in rm
  for (size_t v = 0; v < rm.positionDescriptor.count; v++) {
    vertices.emplace_back(point3(rm.positions[v * 3 + 0], rm.positions[v * 3 + 1],
                                 rm.positions[v * 3 + 2]));
    // TODO : read normals
  }
  typedef std::pair<std::pair<size_t, size_t>, size_t> faceKey;
  std::map<std::pair<size_t, size_t>, size_t> edgesMap;
  std::map<faceKey, size_t> facesMap;
  auto edgeVerticesSorted = [](size_t a,
                               size_t b) -> std::pair<size_t, size_t> {
    return std::pair<size_t, size_t>(std::min(a, b), std::max(a, b));
  };
  auto makeFaceKey = [](size_t a, size_t b, size_t c) -> faceKey {
    if (a < b && a < c)
      return faceKey(std::pair<size_t, size_t>(a, b), c);
    if (b < a && b < c)
      return faceKey(std::pair<size_t, size_t>(b, c), a);
    return faceKey(std::pair<size_t, size_t>(c, a), b);
  };
  auto addFace = [&](size_t a, size_t b, size_t c) -> size_t {
    auto k1 = makeFaceKey(a, b, c);
    if (facesMap.find(k1) == facesMap.end()) {
      facesMap[k1] = faces.size();
      faces.emplace_back();
      size_t faceId = faces.size() - 1;
      faces[faceId].edges[0] = edgesMap[edgeVerticesSorted(a, b)];
      edges[faces[faceId].edges[0]].face = faceId;
      faces[faceId].edges[1] = edgesMap[edgeVerticesSorted(b, c)];
      edges[faces[faceId].edges[1]].face = faceId;
      faces[faceId].edges[2] = edgesMap[edgeVerticesSorted(c, a)];
      edges[faces[faceId].edges[2]].face = faceId;
    } else FATAL_ASSERT(0);
    auto it = facesMap.find(makeFaceKey(a, c, b));
    if (it != facesMap.end()) {
      faces[faces.size() - 1].hface = it->second;
      faces[it->second].hface = faces.size() - 1;
    }
    return faces.size() - 1u;
  };
  for (size_t t = 0; t < rm.meshDescriptor.count; t++) {
    // get vertices from tetrahedron and build faces
    size_t vsi[4]; // vertices indices
    point3 vs[4];  // vertices points
    for (size_t v = 0; v < 4; v++) {
      vsi[v] = static_cast<size_t>(rm.indices[t * 4 + v].positionIndex);
      vs[v] = rm.positionElement(t, v);
    }
    // add edges
    for (size_t e1 = 0; e1 < 3; e1++)
      for (size_t e2 = e1 + 1; e2 < 4; e2++) {
        auto ei = edgeVerticesSorted(vsi[e1], vsi[e2]);
        if (edgesMap.find(ei) == edgesMap.end()) {
          edgesMap[ei] = edges.size();
          edges.emplace_back(ei.first, ei.second);
          vertices[ei.first].edges.insert(edges.size() - 1u);
          vertices[ei.second].edges.insert(edges.size() - 1u);
        }
      }
    // test orientation
    // if true, the first 3 are on the right order, otherwise we must invert
    if (0.f > triple(vs[1] - vs[0], vs[2] - vs[0], vs[3] - vs[0]))
      std::swap(vsi[1], vsi[2]);
    // add faces (0,1,2) (0,3,1) (1,3,2) (0,2,3)
    // faces must start with the vertex of smallest index
    // face 0 1 2
    addFace(vsi[0], vsi[1], vsi[2]);
    // face 0 3 1
    addFace(vsi[0], vsi[3], vsi[1]);
    // face 1 3 2
    addFace(vsi[1], vsi[3], vsi[2]);
    // face 0 2 3
    addFace(vsi[0], vsi[2], vsi[3]);
    // connect neighbors, each face is neighbor of all other 3
    faces[faces.size() - 4].neighbours[0] = faces.size() - 3;
    faces[faces.size() - 4].neighbours[1] = faces.size() - 2;
    faces[faces.size() - 4].neighbours[2] = faces.size() - 1;
    faces[faces.size() - 3].neighbours[0] = faces.size() - 4;
    faces[faces.size() - 3].neighbours[1] = faces.size() - 2;
    faces[faces.size() - 3].neighbours[2] = faces.size() - 1;
    faces[faces.size() - 2].neighbours[0] = faces.size() - 3;
    faces[faces.size() - 2].neighbours[1] = faces.size() - 4;
    faces[faces.size() - 2].neighbours[2] = faces.size() - 1;
    faces[faces.size() - 1].neighbours[0] = faces.size() - 3;
    faces[faces.size() - 1].neighbours[1] = faces.size() - 2;
    faces[faces.size() - 1].neighbours[2] = faces.size() - 4;
    for (size_t f = faces.size() - 4; f < faces.size(); f++)
      faces[f].face3 = tetrahedron.size();
    // add tetrahedron
    tetrahedron.emplace_back(faces.size() - 1);
  }
}

template<typename V, typename F, typename E, typename P>
std::vector<size_t> TMesh<V, F, E, P>::tetrahedronVertices(size_t t) const {
  // get any two faces
  std::set<int> s;
  int f[2];
  f[0] = tetrahedron[t].face;
  f[1] = faces[f[0]].neighbours[0];
  for (int i = 0; i < 2; i++)
    for (auto e : faces[f[i]].edges) {
      s.insert(edges[e].a);
      s.insert(edges[e].b);
    }
  FATAL_ASSERT(s.size() == 4u);
  std::vector<size_t> r;
  for (auto v : s)
    r.emplace_back(v);
  return r;
}

template<typename V, typename F, typename E, typename P>
std::vector<size_t> TMesh<V, F, E, P>::tetrahedronFaces(size_t t) const {
  std::vector<size_t> r = {tetrahedron[t].face,
                           static_cast<size_t>(faces[tetrahedron[t].face].neighbours[0]),
                           static_cast<size_t>(faces[tetrahedron[t].face].neighbours[1]),
                           static_cast<size_t>(faces[tetrahedron[t].face].neighbours[2])};
  return r;
}

template<typename V, typename F, typename E, typename P>
std::vector<size_t> TMesh<V, F, E, P>::faceVertices(size_t f) const {
  auto e0 = edges[faces[f].edges[0]];
  auto e1 = edges[faces[f].edges[1]];
  // retrieve vertices order
  if (e0.a == e1.a)
    return std::vector<size_t>({e0.b, e0.a, e1.b});
  if (e0.a == e1.b)
    return std::vector<size_t>({e0.b, e0.a, e1.a});
  if (e0.b == e1.a)
    return std::vector<size_t>({e0.a, e0.b, e1.b});
  return std::vector<size_t>({e0.a, e0.b, e1.a});
}

template<typename V, typename F, typename E, typename P>
std::vector<size_t> TMesh<V, F, E, P>::tetrahedronNeighbours(size_t t) const {
  auto fs = tetrahedronFaces(t);
  std::vector<size_t> r;
  for (auto f : fs)
    if (faces[f].hface >= 0 && faces[faces[f].hface].face3 >= 0)
      r.emplace_back(faces[faces[f].hface].face3);
  return r;
}

template<typename V, typename F, typename E, typename P>
std::vector<size_t> TMesh<V, F, E, P>::vertexNeighbours(size_t v) const {
  std::vector<size_t> r;
  for (auto e : vertices[v].edges)
    r.emplace_back((edges[e].a == v) ? edges[e].b : edges[e].a);
  return r;
}

template<typename V, typename F, typename E, typename P>
std::vector<size_t> TMesh<V, F, E, P>::vertexTetrahedra(size_t v) const {
  std::set<int> fs;
  for (auto e : vertices[v].edges) {
    fs.insert(edges[e].face);
    fs.insert(faces[edges[e].face].hface);
  }
  std::set<int> ts;
  for(auto f : fs)
    if(f >= 0 && faces[f].face3 >= 0)
      ts.insert(faces[f].face3);
  std::vector<size_t> r;
  for(auto t : ts)
    r.emplace_back(t);
  return r;
}
