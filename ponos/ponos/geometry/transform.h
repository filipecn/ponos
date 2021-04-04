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

#ifndef PONOS_GEOMETRY_TRANSFORM_H
#define PONOS_GEOMETRY_TRANSFORM_H

#include <ponos/geometry/bbox.h>
#include <ponos/geometry/matrix.h>
#include <ponos/geometry/normal.h>
#include <ponos/geometry/point.h>
#include <ponos/geometry/ray.h>
#include <ponos/geometry/vector.h>
#include <ponos/log/debug.h>
#include <ponos/common/bitmask_operators.h>

namespace ponos {

enum class transform_options {
  none = 0x0,
  x_right = 0x1,
  y_right = 0x2,
  z_right = 0x4,
  left_handed = 0x8,
  x_left = 0x10,
  y_left = 0x20,
  z_left = 0x40,
  right_handed = 0x80,
  x_up = 0x100,
  y_up = 0x200,
  z_up = 0x400,
  zero_to_one = 0x800,
  x_down = 0x1000,
  y_down = 0x2000,
  z_down = 0x4000,
  transpose = 0x8000
};
PONOS_ENABLE_BITMASK_OPERATORS(transform_options);

class Transform2 {
public:
  Transform2() = default;
  Transform2(const mat3 &mat, const mat3 &inv_mat);
  explicit Transform2(const bbox2 &bbox);
  void reset();
  void translate(const vec2 &d);
  void scale(real_t x, real_t y);
  void rotate(real_t angle);
  friend Transform2 inverse(const Transform2 &t);
  void operator()(const point2 &p, point2 *r) const {
    real_t x = p.x, y = p.y;
    r->x = m[0][0] * x + m[0][1] * y + m[0][2];
    r->y = m[1][0] * x + m[1][1] * y + m[1][2];
    real_t wp = m[2][0] * x + m[2][1] * y + m[2][2];
    if (wp != 1.f)
      *r /= wp;
  }
  void operator()(const vec2 &v, vec2 *r) const {
    real_t x = v.x, y = v.y;
    r->x = m[0][0] * x + m[0][1] * y;
    r->y = m[1][0] * x + m[1][1] * y;
  }
  vec2 operator()(const vec2 &v) const {
    real_t x = v.x, y = v.y;
    return vec2(m[0][0] * x + m[0][1] * y, m[1][0] * x + m[1][1] * y);
  }
  point2 operator()(const point2 &p) const {
    real_t x = p.x, y = p.y;
    real_t xp = m[0][0] * x + m[0][1] * y + m[0][2];
    real_t yp = m[1][0] * x + m[1][1] * y + m[1][2];
    real_t wp = m[2][0] * x + m[2][1] * y + m[2][2];
    if (wp == 1.f)
      return point2(xp, yp);
    return point2(xp / wp, yp / wp);
  }
  bbox2 operator()(const bbox2 &b) const {
    const Transform2 &M = *this;
    bbox2 ret;
    ret = make_union(ret, M(point2(b.lower.x, b.lower.y)));
    ret = make_union(ret, M(point2(b.upper.x, b.lower.y)));
    ret = make_union(ret, M(point2(b.upper.x, b.upper.y)));
    ret = make_union(ret, M(point2(b.lower.x, b.upper.y)));
    return ret;
  }
  Transform2 operator*(const Transform2 &t) const {
    mat3 m1 = m * t.m;
    mat3 m1_inv = t.m_inv * m_inv;
    return {m1, m1_inv};
  }
  Ray2 operator()(const Ray2 &r) {
    Ray2 ret = r;
    (*this)(ret.o, &ret.o);
    (*this)(ret.d, &ret.d);
    return ret;
  }
  vec2 getTranslate() const { return vec2(m[0][2], m[1][2]); }
  [[nodiscard]] vec2 getScale() const { return s; }
  void computeInverse() { m_inv = inverse(m); }
  [[nodiscard]] mat3 getMatrix() const { return m; }

private:
  mat3 m, m_inv;
  vec2 s;
};

Transform2 scale(real_t x, real_t y);
Transform2 scale(const vec2 &s);
Transform2 rotate(real_t angle);
Transform2 translate(const vec2 &v);
Transform2 inverse(const Transform2 &t);

class Transform {
public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Transform() = default;
  explicit Transform(const mat4 &mat);
  Transform(const mat4 &mat, const mat4 &inv_mat);
  explicit Transform(const real_t mat[4][4]);
  explicit Transform(const bbox3 &bbox);
  void reset();
  void translate(const vec3 &d);
  void scale(real_t x, real_t y, real_t z);
  friend Transform inverse(const Transform &t);
  // ***********************************************************************
  //                             OPERATORS
  // ***********************************************************************
  bbox3 operator()(const bbox3 &b) const {
    const Transform &M = *this;
    bbox3 ret(M(point3(b.lower.x, b.lower.y, b.lower.z)));
    ret = make_union(ret, M(point3(b.upper.x, b.lower.y, b.lower.z)));
    ret = make_union(ret, M(point3(b.lower.x, b.upper.y, b.lower.z)));
    ret = make_union(ret, M(point3(b.lower.x, b.lower.y, b.upper.z)));
    ret = make_union(ret, M(point3(b.lower.x, b.upper.y, b.upper.z)));
    ret = make_union(ret, M(point3(b.upper.x, b.upper.y, b.lower.z)));
    ret = make_union(ret, M(point3(b.upper.x, b.lower.y, b.upper.z)));
    ret = make_union(ret, M(point3(b.lower.x, b.upper.y, b.upper.z)));
    return ret;
  }
  point3 operator()(const point2 &p) const {
    real_t x = p.x, y = p.y, z = 0.f;
    real_t xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    real_t yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    real_t zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    real_t wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp == 1.f)
      return point3(xp, yp, zp);
    return point3(xp, yp, zp) / wp;
  }
  point3 operator()(const point3 &p) const {
    real_t x = p.x, y = p.y, z = p.z;
    real_t xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    real_t yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    real_t zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    real_t wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp == 1.f)
      return point3(xp, yp, zp);
    return point3(xp, yp, zp) / wp;
  }
  void operator()(const point3 &p, point3 *r) const {
    real_t x = p.x, y = p.y, z = p.z;
    r->x = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    r->y = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    r->z = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    real_t wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp != 1.f)
      *r /= wp;
  }
  vec3 operator()(const vec3 &v) const {
    real_t x = v.x, y = v.y, z = v.z;
    return vec3(m[0][0] * x + m[0][1] * y + m[0][2] * z,
                m[1][0] * x + m[1][1] * y + m[1][2] * z,
                m[2][0] * x + m[2][1] * y + m[2][2] * z);
  }
  normal3 operator()(const normal3 &n) const {
    real_t x = n.x, y = n.y, z = n.z;
    return normal3(m_inv[0][0] * x + m_inv[1][0] * y + m_inv[2][0] * z,
                   m_inv[0][1] * x + m_inv[1][1] * y + m_inv[2][1] * z,
                   m_inv[0][2] * x + m_inv[1][2] * y + m_inv[2][2] * z);
  }
  Ray3 operator()(const Ray3 &r) {
    Ray3 ret = r;
    (*this)(ret.o, &ret.o);
    ret.d = (*this)(ret.d);
    return ret;
  }
  void operator()(const Ray3 &r, Ray3 *ret) const {
    (*this)(r.o, &ret->o);
    ret->d = (*this)(ret->d);
  }
  Transform &operator=(const Transform2 &t) {
    m.setIdentity();
    mat3 m3 = t.getMatrix();
    m[0][0] = m3[0][0];
    m[0][1] = m3[0][1];
    m[0][3] = m3[0][2];

    m[1][0] = m3[1][0];
    m[1][1] = m3[1][1];
    m[1][3] = m3[1][2];

    m_inv = inverse(m);
    return *this;
  }
  Transform operator*(const Transform &t) const {
    mat4 m1 = m * t.m;
    mat4 m1_inv = t.m_inv * m_inv;
    return Transform(m1, m1_inv);
  }
  point3 operator*(const point3 &p) const { return (*this)(p); }
  bool operator==(const Transform &t) const { return t.m == m; }
  bool operator!=(const Transform &t) const { return t.m != m; }
  // ***********************************************************************
  //                              METHODS
  // ***********************************************************************
  /// \return true if this transformation changes the coordinate system
  /// handedness
  [[nodiscard]] bool swapsHandedness() const;
  [[nodiscard]] const real_t *c_matrix() const { return &m[0][0]; }
  [[nodiscard]] const mat4 &matrix() const { return m; }
  [[nodiscard]] mat3 upperLeftMatrix() const {
    return mat3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1],
                m[1][2], m[2][0], m[2][1], m[2][2]);
  }
  [[nodiscard]] vec3 getTranslate() const { return vec3(m[0][3], m[1][3], m[2][3]); }
  void computeInverse() { m_inv = inverse(m); }
  bool isIdentity() { return m.isIdentity(); }
  void applyToPoint(const real_t *p, real_t *r, size_t d = 3) const {
    real_t x = p[0], y = p[1], z = 0.f;
    if (d == 3)
      z = p[2];
    r[0] = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    r[1] = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    if (d == 3)
      r[2] = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    real_t wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp != 1.f) {
      real_t invwp = 1.f / wp;
      r[0] *= invwp;
      r[1] *= invwp;
      if (d == 3)
        r[2] *= invwp;
    }
  }

  // **************************************************************************
  //                           STATIC  METHODS
  // **************************************************************************
  /// Look At Transform
  /// \note This transform is commonly used (in graphics) to orient a camera so
  /// it looks at a certain **target** position from its **eye** position.
  /// Given an **up** vector to define the camera orientation, a new coordinate
  /// basis consisting of three vectors {r, u, v} is defined. Where
  /// v = (eye - target) / ||eye - target||
  /// r = -(v x up) / ||(v x up)||
  /// u = v x r
  /// \note The transform is then composed of a translation (to camera **eye**
  /// position) and a basis transform to align r with (1,0,0), u with (0,1,0)
  /// and v with (0,0,1). The final matrix is
  ///     rx  ry  rz  -dot(t, r)
  ///     rx  ry  rz  -dot(t, u)
  ///     rx  ry  rz  -dot(t, v)
  ///      0   0   0      1
  /// \note Note that this transform is built on a left handed coordinate system.
  /// \param eye camera position
  /// \param target camera target
  /// \param up orientation vector
  /// \param left_handed
  /// \return
  static Transform lookAt(const point3 &eye, const point3 &target = {0, 0, 0},
                          const vec3 &up = {0, 1, 0},
                          transform_options options = transform_options::left_handed);
  /// Orthographic Projection
  /// \note In an orthographic projection, parallel lines remain parallel and objects
  /// maintain the same size regardless the distance.
  /// \note This transform projects points into the cube (-1,-1,-1) x (1, 1, 1). It is
  /// also possible to choose to project to (-1,-1, 0) x (1, 1, 1) with the
  /// zero_to_one option.
  /// \note The matrix takes the form:
  ///     2 / (r - l)       0             0         -(r + l) / (r - l)
  ///         0         2 / (t - b)       0         -(t + b) / (t - b)
  ///         0             0         2 / (f - n)   -(f + n) / (f - n)
  ///         0             0             0                  1
  /// \note In the case of zero_to_one == true, the matrix becomes:
  ///     2 / (r - l)       0             0         -(r + l) / (r - l)
  ///         0         2 / (t - b)       0         -(t + b) / (t - b)
  ///         0             0         1 / (f - n)          n / (f - n)
  ///         0             0             0                  1
  /// \note - Note that n > f. This function negates the values of near and
  /// far in case the given values are f > n. Because by default, this
  /// transform uses a left-handed coordinate system.
  /// \param left
  /// \param right
  /// \param bottom
  /// \param top
  /// \param near
  /// \param far
  /// \param left_handed
  /// \param zero_to_one
  /// \return
  static Transform ortho(real_t left, real_t right, real_t bottom, real_t top,
                         real_t near, real_t far,
                         transform_options options = transform_options::left_handed);
  /// Perspective Projection
  /// \note The perspective projection transforms the view frustrum (a pyramid
  /// truncated by a near plane and a far plane, both orthogonal to the view
  /// direction) into the cube (-1,-1,-1) x (1, 1, 1).
  /// \note In a right-handed coordinate system when x points to the right, z
  /// points forward if y points downward and z points backwards if y points
  /// upwards.
  /// \note In a left-handed coordinate system when x points to the right, z
  /// points forwards if y points upward and z points backward if y points
  /// downwards.
  ///
  /// \note It is also possible to choose to project to (-1,-1, 0) x (1, 1, 1)
  /// with the zero_to_one option.
  /// \param fovy
  /// \param aspect_ratio
  /// \param near
  /// \param far
  /// \param left_handed
  /// \param zero_to_one
  /// \return
  static Transform perspective(real_t fovy_in_degrees,
                               real_t aspect_ratio,
                               real_t near,
                               real_t far,
                               transform_options options = transform_options::left_handed);
  /// \param fovy
  /// \param aspect
  /// \param z_near
  /// \param z_far
  /// \param zero_to_one
  /// \return
  [[deprecated]]
  static Transform perspectiveRH(real_t fovy, real_t aspect, real_t z_near,
                                 real_t z_far, bool zero_to_one = false) {
    const real_t tan_half_fovy = std::tan(RADIANS(fovy / 2.f));
    mat4 m;
    if (zero_to_one) {
      m[0][0] = 1 / (aspect * tan_half_fovy);
      m[1][1] = 1 / tan_half_fovy;
      m[2][2] = z_far / (z_near - z_far);
      m[2][3] = -1;
      m[3][2] = -(z_far * z_near) / (z_far - z_near);
    } else {
      m[0][0] = 1 / (aspect * tan_half_fovy);
      m[1][1] = 1 / (tan_half_fovy);
      m[2][2] = -(z_far + z_near) / (z_far - z_near);
      m[2][3] = -1;
      m[3][2] = -(2 * z_far * z_near) / (z_far - z_near);
    }
    return Transform(m, inverse(m));
  }
  /// \param pos
  /// \param target
  /// \param up
  /// \return
  [[deprecated]]
  static Transform lookAtRH(const point3 &pos,
                            const point3 &target,
                            const vec3 &up) {
    vec3 f = normalize(target - pos);
    vec3 s = normalize(cross(f, normalize(up)));
    vec3 u = cross(s, f);
    real_t m[4][4];
    m[0][0] = s.x;
    m[1][0] = s.y;
    m[2][0] = s.z;
    m[3][0] = -dot(s, vec3(pos - point3()));

    m[0][1] = u.x;
    m[1][1] = u.y;
    m[2][1] = u.z;
    m[3][1] = -dot(u, vec3(pos - point3()));

    m[0][2] = -f.x;
    m[1][2] = -f.y;
    m[2][2] = -f.z;
    m[3][2] = dot(f, vec3(pos - point3()));

    m[0][3] = 0;
    m[1][3] = 0;
    m[2][3] = 0;
    m[3][3] = 1;

    mat4 cam_to_world(m);
    return Transform(cam_to_world, inverse(cam_to_world));
  }
protected:
  mat4 m, m_inv;
};

Transform segmentToSegmentTransform(point3 a, point3 b, point3 c, point3 d);
Transform inverse(const Transform &t);
Transform translate(const vec3 &d);
Transform scale(real_t x, real_t y, real_t z);
Transform rotateX(real_t angle);
Transform rotateY(real_t angle);
Transform rotateZ(real_t angle);
Transform rotate(real_t angle, const vec3 &axis);
// Same as OpenGL convention
[[deprecated]]
Transform frustumTransform(real_t left, real_t right, real_t bottom, real_t top,
                           real_t near, real_t far);
[[deprecated]]
Transform perspective(real_t fovy, real_t aspect, real_t zNear, real_t zFar);
[[deprecated]]
Transform perspective(real_t fovy, real_t zNear, real_t zFar);
[[deprecated]]
Transform lookAt(const point3 &pos, const point3 &target, const vec3 &up);
[[deprecated]]
Transform lookAtRH(const point3 &pos, const point3 &target, const vec3 &up);
[[deprecated]]
Transform ortho(real_t left, real_t right, real_t bottom, real_t top,
                real_t near = -1.f, real_t far = 1.f);
[[deprecated]]
Transform orthographic(real_t znear, real_t zfar);
} // namespace ponos

#endif
