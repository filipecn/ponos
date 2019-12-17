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

#ifndef PONOS_GEOMETRY_UTILS_H
#define PONOS_GEOMETRY_UTILS_H

#include <ponos/numeric/numeric.h>
#include <ponos/geometry/vector.h>

namespace ponos {

class Geometry {
public:
  /// Converts spherical coordinates into Euclidian coordinates represented by a
  /// vector
  /// \tparam T data type
  /// \param sinTheta sine value of sin(theta)
  /// \param cosTheta cosine value of cos(theta)
  /// \param phi angle in radians
  /// \return direction vector of the given spherical coordinates
  static vec3f sphericalDirection(real_t sinTheta, real_t cosTheta, real_t phi);
  /// Converts spherical coordinates into Euclidian coordinates represented by a
  /// vector with respect to a coordinate frame defined by axes **x, y and z**
  /// \tparam T data type
  /// \param sinTheta sine value of sin(theta)
  /// \param cosTheta cosine value of cos(theta)
  /// \param phi angle in radians
  /// \param x x axis
  /// \param y y axis
  /// \param z z axis
  /// \return direction vector of the given spherical coordinates
  static vec3f sphericalDirection(real_t sinTheta, real_t cosTheta, real_t phi,
                                  const vec3f &x, const vec3f &y,
                                  const vec3f &z);
  /// Converts a direction to spherical angle theta
  /// \param v normalized direction vector
  /// \return value of theta in radians
  static real_t sphericalTheta(const vec3f& v);
  /// Converts a direction to spherical angle phi
  /// \param v normalized direction vector
  /// \return value of phi in radians
  static real_t sphericalPhi(const vec3f& v);
};

/* local coordinate system
 * @v1  base vector
 * @v2  receives the second axis
 * @v3  receives the third axis
 *
 * Construct a local coordinate system given only a single vector.
 *
 * Assumes that **v1** is already normalized.
 */
inline void makeCoordinateSystem(const vec3 &v1, vec3 *v2, vec3 *v3) {
  if (fabsf(v1.x) > fabsf(v1.y)) {
    float invLen = 1.f / sqrtf(v1.x * v1.x + v1.z * v1.z);
    *v2 = vec3(-v1.z * invLen, 0.f, v1.x * invLen);
  } else {
    float invLen = 1.f / sqrtf(v1.y * v1.y + v1.z * v1.z);
    *v2 = vec3(0.f, v1.z * invLen, -v1.y * invLen);
  }
  *v3 = cross(v1, *v2);
}
/* spherical coordinate
 * @w **[in]** vector
 * In spherical coordinates **(theta, phi)**. **Theta** is the given direction
 * to the z axis and **phi** is the angle formed with the x axis after
 * projection onto the **xy** plane.
 * @return cosine of **theta**
 */
inline float cosTheta(const vec3 &w) { return w.z; }
/* spherical coordinate
 * @w **[in]** vector
 * In spherical coordinates **(theta, phi)**. **Theta** is the given direction
 * to the z axis and **phi** is the angle formed with the x axis after
 * projection onto the **xy** plane.
 * @return absolute value of cosine of **theta**
 */
inline float absCosTheta(const vec3 &w) { return fabs(w.z); }
/* spherical coordinate
 * @w **[in]** vector
 * In spherical coordinates **(theta, phi)**. **Theta** is the given direction
 * to the z axis and **phi** is the angle formed with the x axis after
 * projection onto the **xy** plane.
 * @return squared sine of **theta**
 */
inline float sinTheta2(const vec3 &w) {
  return std::max(0.f, 1.f - cosTheta(w) * cosTheta(w));
}
/* spherical coordinate
 * @w **[in]** vector
 * In spherical coordinates **(theta, phi)**. **Theta** is the given direction
 * to the z axis and **phi** is the angle formed with the x axis after
 * projection onto the **xy** plane.
 * @return sine of **theta**
 */
inline float sinTheta(const vec3 &w) { return sqrtf(sinTheta2(w)); }
/* spherical coordinate
 * @w **[in]** vector
 * In spherical coordinates **(theta, phi)**. **Theta** is the given direction
 * to the z axis and **phi** is the angle formed with the x axis after
 * projection onto the **xy** plane.
 * @return cosine of **phi**
 */
inline float cosPhi(const vec3 &w) {
  float sintheta = sinTheta(w);
  if (sintheta == 0.f)
    return 1.f;
  return clamp(w.x / sintheta, -1.f, 1.f);
}
/* spherical coordinate
 * @w **[in]** vector
 * In spherical coordinates **(theta, phi)**. **Theta** is the given direction
 * to the z axis and **phi** is the angle formed with the x axis after
 * projection onto the **xy** plane.
 * @return sine of **phi**
 */
inline float sinPhi(const vec3 &w) {
  float sintheta = sinTheta(w);
  if (sintheta == 0.f)
    return 0.f;
  return clamp(w.z / sintheta, -1.f, 1.f);
}
/* spherical coordinate
 * @w **[in]** vector
 * @return fliped z coordinate
 */
inline vec3 otherHemisphere(const vec3 &w) { return vec3(w.x, w.y, -w.z); }

} // namespace ponos

#endif // PONOS_GEOMETRY_UTILS_H
