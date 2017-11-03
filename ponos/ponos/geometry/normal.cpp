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

#include <ponos/geometry/normal.h>
#include <ponos/geometry/vector.h>

namespace ponos {

Normal2D::Normal2D(float _x, float _y) : x(_x), y(_y) {}

Normal2D::Normal2D(const Vector2 &v) : x(v.x), y(v.y) {}

Normal2D::operator Vector2() const { return Vector2(x, y); }

Vector2 reflect(const Vector2 &a, const Normal2D &n) {
  return a - 2 * dot(a, Vector2(n)) * Vector2(n);
}

Vector2 project(const Vector2 &v, const Normal2D &n) {
  return v - dot(v, vec2(n)) * vec2(n);
}

Normal::Normal(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

Normal::Normal(const Vector3 &v) : x(v.x), y(v.y), z(v.z) {}

Normal::operator Vector3() const { return Vector3(x, y, z); }

Vector3 Normal::reflect(const Vector3 &v) { return ponos::reflect(v, *this); }

Vector3 Normal::project(const Vector3 &v) { return ponos::project(v, *this); }

void Normal::tangential(Vector3 &a, Vector3 &b) {
  ponos::tangential(Vector3(x, y, z), a, b);
}

Vector3 reflect(const Vector3 &a, const Normal &n) {
  return a - 2 * dot(a, Vector3(n)) * Vector3(n);
}

Vector3 project(const Vector3 &v, const Normal &n) {
  return v - dot(v, vec3(n)) * vec3(n);
}

} // ponos namespacec
