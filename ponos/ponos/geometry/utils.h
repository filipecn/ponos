#ifndef PONOS_GEOMETRY_UTILS_H
#define PONOS_GEOMETRY_UTILS_H

#include <ponos/geometry/numeric.h>
#include <ponos/geometry/vector.h>

namespace ponos {

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

} // ponos namespace

#endif // PONOS_GEOMETRY_UTILS_H
