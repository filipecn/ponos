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

#include <ponos/geometry/queries.h>
#include <ponos/geometry/segment.h>
#include <ponos/structures/mesh.h>

namespace ponos {

Mesh::Mesh(const ponos::RawMesh *m, const ponos::Transform &t) {
  mesh.reset(m);
  transform = t;
  bbox = t(m->bbox);
}

bool Mesh::intersect(const ponos::point3 &p) {
  ponos::Transform inv = ponos::inverse(transform);
  ponos::point3 pp = inv(p);
  ponos::Vector<real_t, 3> P(pp.x, pp.y, pp.z);
  // ponos::Ray3 r(inv(p), ponos::vec3(0, 1, 0));
  int hitCount = 0;
  if (P >= ponos::Vector<real_t, 3>(-0.5f) && P <= ponos::Vector<real_t, 3>(0.5f))
    hitCount = 1;
  return hitCount % 2;
}

const ponos::bbox3 &Mesh::getBBox() const { return bbox; }

const ponos::RawMesh *Mesh::getMesh() const { return mesh.get(); }

const ponos::Transform &Mesh::getTransform() const { return transform; }

Mesh2D::Mesh2D(const ponos::RawMesh *m, const ponos::Transform2 &t) {
  mesh.reset(m);
  transform = t;
  bbox = t(bbox2(m->bbox.lower.xy(), m->bbox.upper.xy()));
}

bool Mesh2D::intersect(const ponos::point2 &p) {
  ponos::Transform2 inv = ponos::inverse(transform);
  ponos::point2 pp = inv(p);
  ponos::Vector<real_t, 2> P(pp.x, pp.y);
  int hitCount = 0;
  for (size_t i = 0; i < mesh->meshDescriptor.count; i++) {
    ponos::point2 a(
        mesh->positions[mesh->indices[i * mesh->meshDescriptor.elementSize + 0]
                                .positionIndex *
                            2 +
                        0],
        mesh->positions[mesh->indices[i * mesh->meshDescriptor.elementSize + 0]
                                .positionIndex *
                            2 +
                        1]);
    ponos::point2 b(
        mesh->positions[mesh->indices[i * mesh->meshDescriptor.elementSize + 1]
                                .positionIndex *
                            2 +
                        0],
        mesh->positions[mesh->indices[i * mesh->meshDescriptor.elementSize + 1]
                                .positionIndex *
                            2 +
                        1]);
    if (ray_segment_intersection(Ray2(pp, vec2(1, 0)), Segment2(a, b)))
      hitCount++;
  }
  return hitCount % 2;
}

const ponos::bbox2 &Mesh2D::getBBox() const { return bbox; }

const ponos::RawMesh *Mesh2D::getMesh() const { return mesh.get(); }

const ponos::Transform2 &Mesh2D::getTransform() const { return transform; }

} // namespace ponos
