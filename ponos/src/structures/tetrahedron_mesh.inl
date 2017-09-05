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

#include <map>

template <typename V, typename F, typename E, typename P>
TMesh<V, F, E, P>::TMesh(const RawMesh &rm) {
  ASSERT_FATAL(rm.meshDescriptor.elementSize == 4);
  ASSERT_FATAL(rm.vertexDescriptor.elementSize == 3);
  ASSERT_FATAL(rm.primitiveType == GeometricPrimitiveType::TETRAHEDRA);
  // add vertices
  for (size_t v = 0; v < rm.vertexDescriptor.count; v++) {
    vertices.emplace_back(Point3(rm.vertices[v * 3 + 0], rm.vertices[v * 3 + 1],
                                 rm.vertices[v * 3 + 2]));
    // TODO : read normals
  }
  std::map<std::pair<size_t, size_t>, size_t> edgesMap;
  for (size_t t = 0; t < rm.meshDescriptor.count; t++) {
    // get vertices from tetrahedron and build faces
    size_t vsi[4];
    Point3 vs[4];
    for (size_t v = 0; v < 4; v++) {
      vsi[v] = rm.indices[t * 4 + v].vertexIndex;
      vs[v] = Point3(rm.vertices[vsi[v] + 0], rm.vertices[vsi[v] + 1],
                     rm.vertices[vsi[v] + 2]);
    }
    // test orientation
    // bool orientation =
    //    0.f < triple(vs[1] - vs[0], vs[2] - vs[0], vs[3] - vs[0]);
    // add edges
    // for (size_t e1 = 0; e1 < 3; e1++)
    //  for (size_t e2 = e1 + 1; e2 < 4; e2++) {
    //  }
  }
}
