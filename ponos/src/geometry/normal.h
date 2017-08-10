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

#ifndef PONOS_GEOMETRY_NORMAL_H
#define PONOS_GEOMETRY_NORMAL_H

#include <iostream>

namespace ponos {

class Vector2;
class Normal2D;
/** \brief  reflects **a** on **n**
 * \param a vector to be reflected
 * \param n axis of reflection
 * \returns reflected **a**
 */
Vector2 reflect(const Vector2 &a, const Normal2D &n);
/** \brief projects **v** on the surface with normal **n**
 * \param v vector
 * \param n surface's normal
 * \returns projected **v**
 */
Vector2 project(const Vector2 &v, const Normal2D &n);

class Normal2D {
public:
  explicit Normal2D(float _x, float _y);
  explicit Normal2D(const Vector2 &v);
  explicit operator Vector2() const;
  Normal2D() { x = y = 0.; }

  Normal2D operator-() const { return Normal2D(-x, -y); }
  Normal2D &operator*=(float f) {
    x *= f;
    y *= f;
    return *this;
  }
  friend std::ostream &operator<<(std::ostream &os, const Normal2D &n) {
    os << "[Normal] " << n.x << " " << n.y << std::endl;
    return os;
  }
  float x, y;
};

class Vector3;
class Normal;
/** \brief  reflects **a** on **n**
 * \param a vector to be reflected
 * \param n axis of reflection
 * \returns reflected **a**
 */
Vector3 reflect(const Vector3 &a, const Normal &n);
/** \brief projects **v** on the surface with normal **n**
 * \param v vector
 * \param n surface's normal
 * \returns projected **v**
 */
Vector3 project(const Vector3 &v, const Normal &n);
/** normal vector */
class Normal {
public:
  explicit Normal(float _x, float _y, float _z);
  explicit Normal(const Vector3 &v);
  explicit operator Vector3() const;
  Normal() { x = y = z = 0.; }

  Normal operator-() const { return Normal(-x, -y, -z); }
  Normal &operator*=(float f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
  }

  /** \brief  reflects **v** from this
   * \param v vector to be reflected
   * \returns reflected **v**
   */
  Vector3 reflect(const Vector3 &v);
  /** \brief projects **v** on the surface with this normal
   * \param v vector
   * \returns projected **v**
   */
  Vector3 project(const Vector3 &v);
  /** \brief compute the two orthogonal-tangential vectors from this
   * \param a **[out]** first tangent
   * \param b **[out]** second tangent
   */
  void tangential(Vector3 &a, Vector3 &b);

  friend std::ostream &operator<<(std::ostream &os, const Normal &n) {
    os << "[Normal] " << n.x << " " << n.y << " " << n.z << std::endl;
    return os;
  }
  float x, y, z;
};

//  inline Normal faceForward(const Normal& n, const Vector3& v) {
//    return (dot(n, v) < 0.f) ? -n : n;
//  }

} // ponos namespace

#endif
