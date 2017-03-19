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

#include "helpers/geometry_drawers.h"

namespace aergia {

void fill_box(const ponos::Point2 &a, const ponos::Point2 &b) {
  glBegin(GL_QUADS);
  glVertex(ponos::Point2(a.x, a.y));
  glVertex(ponos::Point2(b.x, a.y));
  glVertex(ponos::Point2(b.x, b.y));
  glVertex(ponos::Point2(a.x, b.y));
  glEnd();
}

void draw_bbox(const ponos::BBox2D &bbox, float *fillColor) {
  glBegin(GL_LINE_LOOP);
  glVertex(ponos::Point2(bbox.pMin.x, bbox.pMin.y));
  glVertex(ponos::Point2(bbox.pMax.x, bbox.pMin.y));
  glVertex(ponos::Point2(bbox.pMax.x, bbox.pMax.y));
  glVertex(ponos::Point2(bbox.pMin.x, bbox.pMax.y));
  glEnd();
  if (fillColor) {
    glColor4fv(fillColor);
    glBegin(GL_QUADS);
    glVertex(ponos::Point2(bbox.pMin.x, bbox.pMin.y));
    glVertex(ponos::Point2(bbox.pMax.x, bbox.pMin.y));
    glVertex(ponos::Point2(bbox.pMax.x, bbox.pMax.y));
    glVertex(ponos::Point2(bbox.pMin.x, bbox.pMax.y));
    glEnd();
  }
}

void draw_bbox(const ponos::BBox &bbox) {
  glBegin(GL_LINE_LOOP);
  glVertex(ponos::Point3(bbox.pMin.x, bbox.pMin.y, bbox.pMin.z));
  glVertex(ponos::Point3(bbox.pMax.x, bbox.pMin.y, bbox.pMin.z));
  glVertex(ponos::Point3(bbox.pMax.x, bbox.pMax.y, bbox.pMin.z));
  glVertex(ponos::Point3(bbox.pMin.x, bbox.pMax.y, bbox.pMin.z));
  glEnd();
  glBegin(GL_LINE_LOOP);
  glVertex(ponos::Point3(bbox.pMin.x, bbox.pMin.y, bbox.pMax.z));
  glVertex(ponos::Point3(bbox.pMax.x, bbox.pMin.y, bbox.pMax.z));
  glVertex(ponos::Point3(bbox.pMax.x, bbox.pMax.y, bbox.pMax.z));
  glVertex(ponos::Point3(bbox.pMin.x, bbox.pMax.y, bbox.pMax.z));
  glEnd();
  glBegin(GL_LINES);
  glVertex(ponos::Point3(bbox.pMin.x, bbox.pMin.y, bbox.pMin.z));
  glVertex(ponos::Point3(bbox.pMin.x, bbox.pMin.y, bbox.pMax.z));

  glVertex(ponos::Point3(bbox.pMax.x, bbox.pMin.y, bbox.pMin.z));
  glVertex(ponos::Point3(bbox.pMax.x, bbox.pMin.y, bbox.pMax.z));

  glVertex(ponos::Point3(bbox.pMax.x, bbox.pMax.y, bbox.pMin.z));
  glVertex(ponos::Point3(bbox.pMax.x, bbox.pMax.y, bbox.pMax.z));

  glVertex(ponos::Point3(bbox.pMin.x, bbox.pMax.y, bbox.pMin.z));
  glVertex(ponos::Point3(bbox.pMin.x, bbox.pMax.y, bbox.pMax.z));

  glEnd();
}

void draw_segment(ponos::Segment3 segment) {
  glBegin(GL_LINES);
  glVertex(segment.a);
  glVertex(segment.b);
  glEnd();
}

void draw_circle(const ponos::Circle &circle,
                 const ponos::Transform2D *transform) {
  glBegin(GL_TRIANGLE_FAN);
  if (transform != nullptr)
    glVertex((*transform)(circle.c));
  else
    glVertex(circle.c);
  float angle = 0.0;
  float step = PI_2 / 100.f;
  while (angle < PI_2 + step) {
    ponos::vec2 pp(circle.r * cosf(angle), circle.r * sinf(angle));
    if (transform != nullptr)
      glVertex((*transform)(circle.c + pp));
    else
      glVertex(circle.c + pp);
    angle += step;
  }
  glEnd();
}

void draw_sphere(ponos::Sphere sphere, const ponos::Transform *transform) {
  if (transform) {
    glPushMatrix();
    glApplyTransform(*transform);
  }
  const float vStep = PI / 20.f;
  const float hStep = PI_2 / 20.f;
  glBegin(GL_TRIANGLES);
  // south pole
  ponos::Point3 pole(0.f, -sphere.r, 0.f);
  for (float angle = 0.f; angle < PI_2; angle += hStep) {
    float r = sphere.r * sinf(vStep);
    glVertex(sphere.c + ponos::vec3(pole));
    glVertex(sphere.c + ponos::vec3(pole) +
             r * ponos::vec3(cosf(angle), -sinf(vStep), sinf(angle)));
    glVertex(sphere.c + ponos::vec3(pole) +
             r * ponos::vec3(cosf(angle + hStep), -sinf(vStep),
                             sinf(angle + hStep)));
  }
  // north pole
  pole = ponos::Point3(0.f, sphere.r, 0.f);
  for (float angle = 0.f; angle < PI_2; angle += hStep) {
    float r = sphere.r * sinf(vStep);
    glVertex(sphere.c + ponos::vec3(pole));
    glVertex(sphere.c + ponos::vec3(pole) +
             r * ponos::vec3(cosf(angle), -sinf(vStep), sinf(angle)));
    glVertex(sphere.c + ponos::vec3(pole) +
             r * ponos::vec3(cosf(angle + hStep), -sinf(vStep),
                             sinf(angle + hStep)));
  }

  glEnd();
  glBegin(GL_QUADS);
  for (float vAngle = vStep; vAngle <= PI - vStep; vAngle += vStep) {
    float r = sphere.r * sinf(vAngle);
    float R = sphere.r * sinf(vAngle + vStep);
    for (float angle = 0.f; angle < PI_2; angle += hStep) {
      glVertex(sphere.c + ponos::vec3(r * cosf(angle), sphere.r * cosf(vAngle),
                                      r * sinf(angle)));
      glVertex(sphere.c + ponos::vec3(r * cosf(angle + hStep),
                                      sphere.r * cosf(vAngle),
                                      r * sinf(angle + hStep)));
      glVertex(sphere.c + ponos::vec3(R * cosf(angle + hStep),
                                      sphere.r * cosf(vAngle + vStep),
                                      R * sinf(angle + hStep)));
      glVertex(sphere.c + ponos::vec3(R * cosf(angle),
                                      sphere.r * cosf(vAngle + vStep),
                                      R * sinf(angle)));
    }
  }
  glEnd();
  if (transform)
    glPopMatrix();
}

void draw_polygon(const ponos::Polygon &polygon,
                  const ponos::Transform2D *transform) {
  glBegin(GL_LINE_LOOP);
  for (const auto &p : polygon.vertices) {
    if (transform != nullptr)
      glVertex((*transform)(p));
    else
      glVertex(p);
  }
  glEnd();
}

void draw_mesh(const ponos::Mesh2D *m, const ponos::Transform2D *t) {
  glLineWidth(3.f);
  const ponos::RawMesh *rm = m->getMesh();
  glBegin(GL_LINES);
  for (size_t i = 0; i < rm->meshDescriptor.count; i++) {
    ponos::Point2 a(
        rm->vertices[rm->indices[i * rm->meshDescriptor.elementSize + 0]
                             .vertexIndex *
                         2 +
                     0],
        rm->vertices[rm->indices[i * rm->meshDescriptor.elementSize + 0]
                             .vertexIndex *
                         2 +
                     1]);
    ponos::Point2 b(
        rm->vertices[rm->indices[i * rm->meshDescriptor.elementSize + 1]
                             .vertexIndex *
                         2 +
                     0],
        rm->vertices[rm->indices[i * rm->meshDescriptor.elementSize + 1]
                             .vertexIndex *
                         2 +
                     1]);
    if (t)
      glVertex((*t)(a));
    else
      glVertex(a);
    if (t)
      glVertex((*t)(b));
    else
      glVertex(b);
  }
  glEnd();
  glLineWidth(1.f);
}

} // aergia namespace
