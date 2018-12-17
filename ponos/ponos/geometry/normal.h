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
#include <ponos/common/defs.h>

namespace ponos {

template <typename T> class Vector2;
template <typename T> class Normal2;
/** \brief  reflects **a** on **n**
 * \param a vector to be reflected
 * \param n axis of reflection
 * \returns reflected **a**
 */
template <typename T>
Vector2<T> reflect(const Vector2<T> &a, const Normal2<T> &n);
/** \brief projects **v** on the surface with normal **n**
 * \param v vector
 * \param n surface's normal
 * \returns projected **v**
 */
template <typename T>
Vector2<T> project(const Vector2<T> &v, const Normal2<T> &n);

template <typename T> class Normal2 {
public:
  explicit Normal2(T _x, T _y);
  explicit Normal2(const Vector2<T> &v);
  explicit operator Vector2<T>() const;
  Normal2() { x = y = 0.; }

  Normal2 operator-() const { return Normal2(-x, -y); }
  Normal2 &operator*=(T f) {
    x *= f;
    y *= f;
    return *this;
  }
  friend std::ostream &operator<<(std::ostream &os, const Normal2 &n) {
    os << "[Normal3] " << n.x << " " << n.y << std::endl;
    return os;
  }
  T x, y;
};

typedef Normal2<real_t> normal2;

template <typename T> class Vector3;
template <typename T> class Normal3;
/** \brief  reflects **a** on **n**
 * \param a vector to be reflected
 * \param n axis of reflection
 * \returns reflected **a**
 */
template <typename T>
Vector3<T> reflect(const Vector3<T> &a, const Normal3<T> &n);
/** \brief projects **v** on the surface with normal **n**
 * \param v vector
 * \param n surface's normal
 * \returns projected **v**
 */
template <typename T>
Vector3<T> project(const Vector3<T> &v, const Normal3<T> &n);
/** normal vector */
template <typename T> class Normal3 {
public:
  explicit Normal3(T _x, T _y, T _z);
  explicit Normal3(const Vector3<T> &v);
  explicit operator Vector3<T>() const;
  Normal3() { x = y = z = 0; }

  Normal3 operator-() const { return Normal3(-x, -y, -z); }
  Normal3 &operator*=(T f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
  }
  bool operator!=(const Normal3 &n) const {
    return n.x != x || n.y != y || n.z != z;
  }

  /** \brief  reflects **v** from this
   * \param v vector to be reflected
   * \returns reflected **v**
   */
  Vector3<T> reflect(const Vector3<T> &v);
  /** \brief projects **v** on the surface with this normal
   * \param v vector
   * \returns projected **v**
   */
  Vector3<T> project(const Vector3<T> &v);
  /** \brief compute the two orthogonal-tangential vectors from this
   * \param a **[out]** first tangent
   * \param b **[out]** second tangent
   */
  void tangential(Vector3<T> &a, Vector3<T> &b);

  friend std::ostream &operator<<(std::ostream &os, const Normal3 &n) {
    os << "[Normal3] " << n.x << " " << n.y << " " << n.z << std::endl;
    return os;
  }
  T x, y, z;
};

typedef Normal3<real_t> normal3;
typedef Normal3<float> normal3f;

template <typename T> Normal3<T> normalize(const Normal3<T> &normal);

#include "normal.inl"

//  inline Normal3 faceForward(const Normal3& n, const Vector3& v) {
//    return (dot(n, v) < 0.f) ? -n : n;
//  }

} // namespace ponos

#endif
