#ifndef AERGIA_HELPERS_GEOMETRY_DRAWERS_H
#define AERGIA_HELPERS_GEOMETRY_DRAWERS_H

#include "utils/open_gl.h"

#include <ponos.h>

namespace aergia {

void fill_box(const ponos::Point2 &a, const ponos::Point2 &b);

void draw_bbox(const ponos::BBox2D &bbox, float *fillColor = nullptr);

void draw_bbox(const ponos::BBox &bbox);

void draw_segment(ponos::Segment3 segment);

void draw_circle(const ponos::Circle &circle,
                 const ponos::Transform2D *transform = nullptr);

void draw_sphere(ponos::Sphere sphere,
                 const ponos::Transform *transform = nullptr);

void draw_polygon(const ponos::Polygon &polygon,
                  const ponos::Transform2D *transform = nullptr);

void draw_mesh(const ponos::Mesh2D *m,
               const ponos::Transform2D *transform = nullptr);

void draw_vector(const ponos::Point2 &p, const ponos::vec2 &v, float w = 0.01f,
                 float h = 0.01f);
} // aergia namespace

#endif // AERGIA_HELPERS_GEOMETRY_DRAWERS_H
