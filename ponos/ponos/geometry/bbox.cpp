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

#include <ponos/geometry/bbox.h>

namespace ponos {

BBox2D::BBox2D() {
  pMin = Point2(INFINITY, INFINITY);
  pMax = Point2(-INFINITY, -INFINITY);
}

BBox2D::BBox2D(const Point2 &p1, const Point2 &p2) {
  pMin = Point2(std::min(p1.x, p2.x), std::min(p1.y, p2.y));
  pMax = Point2(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
}

BBox::BBox() {
  pMin = Point3(INFINITY, INFINITY, INFINITY);
  pMax = Point3(-INFINITY, -INFINITY, -INFINITY);
}

BBox::BBox(const Point3 &p) : pMin(p), pMax(p) {}

BBox::BBox(const Point3 &p1, const Point3 &p2) {
  pMin =
      Point3(std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::min(p1.z, p2.z));
  pMax =
      Point3(std::max(p1.x, p2.x), std::max(p1.y, p2.y), std::max(p1.z, p2.z));
}

} // ponos namespace
