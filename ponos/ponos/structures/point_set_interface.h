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
#include <ponos/geometry/ray.h>

namespace ponos {

class PointSetInterface {
public:
  PointSetInterface() = default;
  virtual ~PointSetInterface() = default;
  /// random access operator
  /// \param i point index
  /// \return position of point **i**
  virtual Point3 operator[](uint i) const = 0;
  /// active points count
  /// \return number of active points
  virtual uint size() = 0;
  /// adds new point with position **p**
  /// \param p position to be added
  /// \return element id, so this position can be accessed later
  virtual uint add(Point3 p) = 0;
  /// sets position **p** to element **i**
  /// \param i element id
  /// \param p new position value
  virtual void setPosition(uint i, Point3 p) = 0;
  /// removes element **i**
  /// \param i element id
  virtual void remove(uint i) = 0;
  /// search points that intersect a bbox. If the internal tree has not been
  /// created it searchs with an implicit tree.
  /// \param b search region, world coordinates
  /// \param f callback to receive the id of each found point
  virtual void search(const BBox &b, const std::function<void(uint)> &f) = 0;
  /// iterate over active points
  /// \param f callback to receive the id and position of each found point
  virtual void
  iteratePoints(const std::function<void(uint, Point3)> &f) const = 0;
  /// \param r ray
  /// \param e the max distance from ray line a point can be intersected
  /// \return -1 if no point is intersect, point id otherwise
  virtual int intersect(const Ray3& r, float e) = 0;
  /// \param r ray
  /// \param f callback to every point **r** intersects
  virtual void cast(const Ray3& r, const std::function<void(uint)>& f) = 0;
};

} // ponos namespace

#endif //PONOS_POINT_SET_INTERFACE_H
