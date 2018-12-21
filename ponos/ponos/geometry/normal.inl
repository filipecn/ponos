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

template <typename T> Normal2<T>::Normal2(T _x, T _y) : x(_x), y(_y) {}

template <typename T>
Normal2<T>::Normal2(const Vector2<T> &v) : x(v.x), y(v.y) {}

template <typename T> Normal2<T>::operator Vector2<T>() const {
  return Vector2<T>(x, y);
}

template <typename T>
Vector2<T> reflect(const Vector2<T> &a, const Normal2<T> &n) {
  return a - 2 * dot(a, Vector2<T>(n)) * Vector2<T>(n);
}

template <typename T>
Vector2<T> project(const Vector2<T> &v, const Normal2<T> &n) {
  return v - dot(v, Vector2<T>(n)) * Vector2<T>(n);
}

template <typename T>
Normal3<T>::Normal3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

template <typename T>
Normal3<T>::Normal3(const Vector3<T> &v) : x(v.x), y(v.y), z(v.z) {}

template <typename T> Normal3<T>::operator Vector3<T>() const {
  return Vector3<T>(x, y, z);
}

template <typename T> Vector3<T> Normal3<T>::reflect(const Vector3<T> &v) {
  return ponos::reflect(v, *this);
}

template <typename T> Vector3<T> Normal3<T>::project(const Vector3<T> &v) {
  return ponos::project(v, *this);
}

template <typename T>
void Normal3<T>::tangential(Vector3<T> &a, Vector3<T> &b) {
  //  ponos::tangential(Vector3<T>(x, y, z), a, b);
}

template <typename T> Normal3<T> normalize(const Normal3<T> &normal) {
  T d = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
  if (d == 0.f)
    return normal;
  return Normal3<T>(normal.x / d, normal.y / d, normal.z / d);
}

template <typename T> Normal3<T> abs(const Normal3<T> &normal) {
  return Normal3<T>(std::abs(normal.x), std::abs(normal.y), std::abs(normal.z));
}

template <typename T>
Vector3<T> reflect(const Vector3<T> &a, const Normal3<T> &n) {
  return a - 2 * dot(a, Vector3<T>(n)) * Vector3<T>(n);
}

template <typename T>
Vector3<T> project(const Vector3<T> &v, const Normal3<T> &n) {
  return v - dot(v, Vector3<T>(n)) * Vector3<T>(n);
}

template <typename T> T dot(const Normal3<T> &n, const Vector3<T> &v) {
  return n.x * v.x + n.y * v.y + n.z * v.z;
}

template <typename T> T dot(const Vector3<T> &v, const Normal3<T> &n) {
  return n.x * v.x + n.y * v.y + n.z * v.z;
}
