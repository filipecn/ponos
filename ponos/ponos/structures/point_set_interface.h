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

#ifndef PONOS_POINT_SET_INTERFACE_H
#define PONOS_POINT_SET_INTERFACE_H

#include <ponos/geometry/bbox.h>

namespace ponos {

class PointSetInterface {
public:
  PointSetInterface() = default;
  virtual ~PointSetInterface() = default;
  virtual Point3 operator[](uint) const = 0;
  virtual uint size() = 0;
  virtual uint add(Point3) = 0;
  virtual void setPosition(uint, Point3) = 0;
  virtual void remove(uint) = 0;
  virtual void search(const BBox &, const std::function<void(uint)> &) = 0;
  //virtual void
  //iteratePoints(const std::function<void(uint, Point3)> &f) const = 0;
};

} // ponos namespace

#endif //PONOS_POINT_SET_INTERFACE_H
