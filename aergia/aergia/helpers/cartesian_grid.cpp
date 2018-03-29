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

#include <aergia/helpers/cartesian_grid.h>

namespace aergia {

CartesianGrid::CartesianGrid(int d) {
  xAxisColor = COLOR_RED;
  yAxisColor = COLOR_BLUE;
  zAxisColor = COLOR_GREEN;
  gridColor = COLOR_BLACK;
  for (auto &plane : planes) {
    plane.low = -d;
    plane.high = d;
  }
}

CartesianGrid::CartesianGrid(int dx, int dy, int dz) {
  planes[0].low = -dx;
  planes[0].high = dx;
  planes[1].low = -dy;
  planes[1].high = dy;
  planes[2].low = -dz;
  planes[2].high = dz;
}

void CartesianGrid::setDimension(size_t d, int a, int b) {
  planes[d].low = a;
  planes[d].high = b;
}

void CartesianGrid::draw() {
  glColor(gridColor);
  glBegin(GL_LINES);
  // XY
  for (int x = planes[0].low; x <= planes[0].high; x++) {
    glVertex(transform(ponos::Point3(1.f * x, 1.f * planes[1].low, 0.f)));
    glVertex(transform(ponos::Point3(1.f * x, 1.f * planes[1].high, 0.f)));
  }
  for (int y = planes[1].low; y <= planes[1].high; y++) {
    glVertex(transform(ponos::Point3(1.f * planes[0].low, 1.f * y, 0.f)));
    glVertex(transform(ponos::Point3(1.f * planes[0].high, 1.f * y, 0.f)));
  }
  // YZ
  for (int y = planes[1].low; y <= planes[1].high; y++) {
    glVertex(transform(ponos::Point3(0.f, 1.f * y, 1.f * planes[2].low)));
    glVertex(transform(ponos::Point3(0.f, 1.f * y, 1.f * planes[2].high)));
  }
  for (int z = planes[2].low; z <= planes[2].high; z++) {
    glVertex(transform(ponos::Point3(0.f, 1.f * planes[1].low, 1.f * z)));
    glVertex(transform(ponos::Point3(0.f, 1.f * planes[1].high, 1.f * z)));
  }
  // XZ
  for (int x = planes[0].low; x <= planes[0].high; x++) {
    glVertex(transform(ponos::Point3(1.f * x, 0.f, 1.f * planes[2].low)));
    glVertex(transform(ponos::Point3(1.f * x, 0.f, 1.f * planes[2].high)));
  }
  for (int z = planes[2].low; z <= planes[2].high; z++) {
    glVertex(transform(ponos::Point3(1.f * planes[1].low, 0.f, 1.f * z)));
    glVertex(transform(ponos::Point3(1.f * planes[1].high, 0.f, 1.f * z)));
  }
  glEnd();
  // axis
  glLineWidth(4.f);
  glBegin(GL_LINES);
  glColor(xAxisColor);
  glVertex(transform(ponos::Point3()));
  glVertex(transform(ponos::Point3(0.5, 0, 0)));
  glColor(yAxisColor);
  glVertex(transform(ponos::Point3()));
  glVertex(transform(ponos::Point3(0, 0.5, 0)));
  glColor(zAxisColor);
  glVertex(transform(ponos::Point3()));
  glVertex(transform(ponos::Point3(0, 0, 0.5)));
  glEnd();
  glLineWidth(1.f);
}

} // aergia namespace
