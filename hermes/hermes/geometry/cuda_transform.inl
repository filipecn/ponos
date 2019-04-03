template <typename T> __host__ __device__ Transform2<T>::Transform2() {
  m.setIdentity();
  m_inv.setIdentity();
}

template <typename T>
__host__ __device__ Transform2<T>::Transform2(const Matrix3x3<T> &mat,
                                              const Matrix3x3<T> inv_mat)
    : m(mat), m_inv(inv_mat) {}

template <typename T>
__host__ __device__ Transform2<T>::Transform2(const BBox2<T> &bbox) {
  m.m[0][0] = bbox.upper[0] - bbox.lower[0];
  m.m[1][1] = bbox.upper[1] - bbox.lower[1];
  m.m[0][2] = bbox.lower[0];
  m.m[1][2] = bbox.lower[1];
  m_inv = inverse(m);
}

template <typename T> __host__ __device__ void Transform2<T>::reset() {
  m.setIdentity();
}

template <typename T>
__host__ __device__ void Transform2<T>::translate(const Vector2<T> &d) {
  // TODO update inverse and make a better implementarion
  UNUSED_VARIABLE(d);
}

template <typename T> __host__ __device__ void Transform2<T>::scale(T x, T y) {
  // TODO update inverse and make a better implementarion
  UNUSED_VARIABLE(x);
  UNUSED_VARIABLE(y);
}

template <typename T> __host__ __device__ void Transform2<T>::rotate(T angle) {
  T sin_a = sinf(TO_RADIANS(angle));
  T cos_a = cosf(TO_RADIANS(angle));
  Matrix3x3<T> M(cos_a, -sin_a, 0.f, sin_a, cos_a, 0.f, 0.f, 0.f, 1.f);
  Vector2<T> t = getTranslate();
  m.m[0][2] = m.m[1][2] = 0;
  m = m * M;
  m.m[0][2] = t.x;
  m.m[1][2] = t.y;
  // TODO update inverse and make a better implementarion
  m_inv = inverse(m);
}

template <typename T> __host__ __device__ Transform2<T> rotate(T angle) {
  T sin_a = sinf(TO_RADIANS(angle));
  T cos_a = cosf(TO_RADIANS(angle));
  Matrix3x3<T> m(cos_a, -sin_a, 0.f, sin_a, cos_a, 0.f, 0.f, 0.f, 1.f);
  return Transform2<T>(m, transpose(m));
}

template <typename T>
__host__ __device__ Transform2<T> translate(const Vector2<T> &v) {
  Matrix3x3<T> m(1.f, 0.f, v.x, 0.f, 1.f, v.y, 0.f, 0.f, 1.f);
  Matrix3x3<T> m_inv(1.f, 0.f, -v.x, 0.f, 1.f, -v.y, 0.f, 0.f, 1.f);
  return Transform2<T>(m, m_inv);
}

template <typename T>
__host__ __device__ Transform2<T> inverse(const Transform2<T> &t) {
  return Transform2<T>(t.m_inv, t.m);
}

template <typename T> __host__ __device__ Transform<T>::Transform() {
  m.setIdentity();
  m_inv.setIdentity();
}

template <typename T>
__host__ __device__ Transform<T>::Transform(const Matrix4x4<T> &mat)
    : m(mat), m_inv(inverse(mat)) {}

template <typename T>
__host__ __device__ Transform<T>::Transform(const Matrix4x4<T> &mat,
                                            const Matrix4x4<T> &inv_mat)
    : m(mat), m_inv(inv_mat) {}

template <typename T>
__host__ __device__ Transform<T>::Transform(const T mat[4][4]) {
  m = Matrix4x4<T>(mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0],
                   mat[1][1], mat[1][2], mat[1][3], mat[2][0], mat[2][1],
                   mat[2][2], mat[2][3], mat[3][0], mat[3][1], mat[3][2],
                   mat[3][3]);
  m_inv = inverse(m);
}

template <typename T>
__host__ __device__ Transform<T>::Transform(const bbox3 &bbox) {
  m.m[0][0] = bbox.upper[0] - bbox.lower[0];
  m.m[1][1] = bbox.upper[1] - bbox.lower[1];
  m.m[2][2] = bbox.upper[2] - bbox.lower[2];
  m.m[0][3] = bbox.lower[0];
  m.m[1][3] = bbox.lower[1];
  m.m[2][3] = bbox.lower[2];
  m_inv = inverse(m);
}

template <typename T> __host__ __device__ void Transform<T>::reset() {
  m.setIdentity();
}

template <typename T>
__host__ __device__ void Transform<T>::translate(const Vector3<T> &d) {
  // TODO update inverse and make a better implementarion
  UNUSED_VARIABLE(d);
}

template <typename T>
__host__ __device__ void Transform<T>::scale(T x, T y, T z) {
  // TODO update inverse and make a better implementarion
  UNUSED_VARIABLE(x);
  UNUSED_VARIABLE(y);
  UNUSED_VARIABLE(z);
}

template <typename T>
__host__ __device__ bool Transform<T>::swapsHandedness() const {
  T det = (m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1])) -
          (m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0])) +
          (m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]));
  return det < 0;
}

template <typename T> __host__ __device__ Transform2<T> scale(T x, T y) {
  Matrix3x3<T> m(x, 0, 0, 0, y, 0, 0, 0, 1);
  Matrix3x3<T> inv(1.f / x, 0, 0, 0, 1.f / y, 0, 0, 0, 1);
  return Transform2<T>(m, inv);
}

template <typename T>
__host__ __device__ Transform<T>
segmentToSegmentTransform(Point3<T> a, Point3<T> b, Point3<T> c, Point3<T> d) {
  // Consider two bases a b e f and c d g h
  // TODO implement
  return Transform<T>();
}

template <typename T>
__host__ __device__ Transform<T> inverse(const Transform<T> &t) {
  return Transform<T>(t.m_inv, t.m);
}

template <typename T>
__host__ __device__ Transform<T> translate(const Vector3<T> &d) {
  Matrix4x4<T> m(1.f, 0.f, 0.f, d.x, 0.f, 1.f, 0.f, d.y, 0.f, 0.f, 1.f, d.z,
                 0.f, 0.f, 0.f, 1.f);
  Matrix4x4<T> m_inv(1.f, 0.f, 0.f, -d.x, 0.f, 1.f, 0.f, -d.y, 0.f, 0.f, 1.f,
                     -d.z, 0.f, 0.f, 0.f, 1.f);
  return Transform<T>(m, m_inv);
}

template <typename T> __host__ __device__ Transform<T> scale(T x, T y, T z) {
  Matrix4x4<T> m(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
  Matrix4x4<T> inv(1.f / x, 0, 0, 0, 0, 1.f / y, 0, 0, 0, 0, 1.f / z, 0, 0, 0,
                   0, 1);
  return Transform<T>(m, inv);
}

template <typename T> __host__ __device__ Transform<T> rotateX(T angle) {
  T sin_a = sinf(TO_RADIANS(angle));
  T cos_a = cosf(TO_RADIANS(angle));
  Matrix4x4<T> m(1.f, 0.f, 0.f, 0.f, 0.f, cos_a, -sin_a, 0.f, 0.f, sin_a, cos_a,
                 0.f, 0.f, 0.f, 0.f, 1.f);
  return Transform<T>(m, transpose(m));
}

template <typename T> __host__ __device__ Transform<T> rotateY(T angle) {
  T sin_a = sinf(TO_RADIANS(angle));
  T cos_a = cosf(TO_RADIANS(angle));
  Matrix4x4<T> m(cos_a, 0.f, sin_a, 0.f, 0.f, 1.f, 0.f, 0.f, -sin_a, 0.f, cos_a,
                 0.f, 0.f, 0.f, 0.f, 1.f);
  return Transform<T>(m, transpose(m));
}

template <typename T> __host__ __device__ Transform<T> rotateZ(T angle) {
  T sin_a = sinf(TO_RADIANS(angle));
  T cos_a = cosf(TO_RADIANS(angle));
  Matrix4x4<T> m(cos_a, -sin_a, 0.f, 0.f, sin_a, cos_a, 0.f, 0.f, 0.f, 0.f, 1.f,
                 0.f, 0.f, 0.f, 0.f, 1.f);
  return Transform<T>(m, transpose(m));
}

template <typename T>
__host__ __device__ Transform<T> rotate(T angle, const Vector3<T> &axis) {
  Vector3<T> a = normalize(axis);
  T s = sinf(TO_RADIANS(angle));
  T c = cosf(TO_RADIANS(angle));
  T m[4][4];

  m[0][0] = a.x * a.x + (1.f - a.x * a.x) * c;
  m[0][1] = a.x * a.y * (1.f - c) - a.z * s;
  m[0][2] = a.x * a.z * (1.f - c) + a.y * s;
  m[0][3] = 0;

  m[1][0] = a.x * a.y * (1.f - c) + a.z * s;
  m[1][1] = a.y * a.y + (1.f - a.y * a.y) * c;
  m[1][2] = a.y * a.z * (1.f - c) - a.x * s;
  m[1][3] = 0;

  m[2][0] = a.x * a.z * (1.f - c) - a.y * s;
  m[2][1] = a.y * a.z * (1.f - c) + a.x * s;
  m[2][2] = a.z * a.z + (1.f - a.z * a.z) * c;
  m[2][3] = 0;

  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;
  m[3][3] = 1;

  Matrix4x4<T> mat(m);
  return Transform<T>(mat, transpose(mat));
}

template <typename T>
__host__ __device__ Transform<T> frustumTransform(T left, T right, T bottom,
                                                  T top, T near, T far) {
  T tnear = 2.f * near;
  T lr = right - left;
  T bt = top - bottom;
  T nf = far - near;
  T m[4][4];
  m[0][0] = tnear / lr;
  m[0][1] = 0.f;
  m[0][2] = (right + left) / lr;
  m[0][3] = 0.f;

  m[1][0] = 0.f;
  m[1][1] = tnear / bt;
  m[1][2] = (top + bottom) / bt;
  m[1][3] = 0.f;

  m[2][0] = 0.f;
  m[2][1] = 0.f;
  m[2][2] = (-far - near) / nf;
  m[2][3] = (-tnear * far) / nf;

  m[3][0] = 0.f;
  m[3][1] = 0.f;
  m[3][2] = -1.f;
  m[3][3] = 0.f;

  Matrix4x4<T> projection(m);
  return Transform<T>(projection, inverse(projection));
}

template <typename T>
__host__ __device__ Transform<T> perspective(T fov, T aspect, T zNear, T zFar) {
  T xmax = zNear * tanf(TO_RADIANS(fov / 2.f));
  T ymax = xmax / aspect;
  return frustumTransform(-xmax, xmax, -ymax, ymax, zNear, zFar);
}

template <typename T>
__host__ __device__ Transform<T> perspective(T fov, T n, T f) {
  // perform projectiev divide
  Matrix4x4<T> persp = Matrix4x4<T>(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, f / (f - n),
                                    -f * n / (f - n), 0, 0, 1, 0);
  // scale to canonical viewing volume
  T invTanAng = 1.f / tanf(TO_RADIANS(fov) / 2.f);
  return scale(invTanAng, invTanAng, 1) * Transform<T>(persp);
}

template <typename T>
__host__ __device__ Transform<T>
lookAt(const Point3<T> &pos, const Point3<T> &target, const Vector3<T> &up) {
  Vector3<T> dir = normalize(target - pos);
  Vector3<T> left = normalize(cross(normalize(up), dir));
  Vector3<T> new_up = cross(dir, left);
  T m[4][4];
  m[0][0] = left.x;
  m[1][0] = left.y;
  m[2][0] = left.z;
  m[3][0] = 0;

  m[0][1] = new_up.x;
  m[1][1] = new_up.y;
  m[2][1] = new_up.z;
  m[3][1] = 0;

  m[0][2] = dir.x;
  m[1][2] = dir.y;
  m[2][2] = dir.z;
  m[3][2] = 0;

  m[0][3] = pos.x;
  m[1][3] = pos.y;
  m[2][3] = pos.z;
  m[3][3] = 1;

  Matrix4x4<T> cam_to_world(m);
  return Transform<T>(inverse(cam_to_world), cam_to_world);
}

template <typename T>
__host__ __device__ Transform<T>
lookAtRH(const Point3<T> &pos, const Point3<T> &target, const Vector3<T> &up) {
  Vector3<T> dir = normalize(pos - target);
  Vector3<T> left = normalize(cross(normalize(up), dir));
  Vector3<T> new_up = cross(dir, left);
  T m[4][4];
  m[0][0] = left.x;
  m[0][1] = left.y;
  m[0][2] = left.z;
  m[0][3] = -dot(left, Vector3<T>(pos - Point3<T>()));

  m[1][0] = new_up.x;
  m[1][1] = new_up.y;
  m[1][2] = new_up.z;
  m[1][3] = -dot(new_up, Vector3<T>(pos - Point3<T>()));

  m[2][0] = dir.x;
  m[2][1] = dir.y;
  m[2][2] = dir.z;
  m[2][3] = -dot(dir, Vector3<T>(pos - Point3<T>()));

  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;
  m[3][3] = 1;

  /*
    dir = normalize(target - pos);
    left = normalize(cross(normalize(up), dir));
    new_up = cross(dir, left);

    m[0][3] = pos.x;
    m[1][3] = pos.y;
    m[2][3] = pos.z;
    m[3][3] = 1;
    m[0][0] = left.x;
    m[1][0] = left.y;
    m[2][0] = left.z;
    m[3][0] = 0;
    m[0][1] = new_up.x;
    m[1][1] = new_up.y;
    m[2][1] = new_up.z;
    m[3][1] = 0;
    m[0][2] = dir.x;
    m[1][2] = dir.y;
    m[2][2] = dir.z;
    m[3][2] = 0;
  */
  Matrix4x4<T> cam_to_world(m);
  return Transform<T>(cam_to_world, inverse(cam_to_world));
}

template <typename T>
__host__ __device__ Transform<T> ortho(T left, T right, T bottom, T top, T near,
                                       T far) {
  T m[4][4];

  m[0][0] = 2.f / (right - left);
  m[1][0] = 0.f;
  m[2][0] = 0.f;
  m[3][0] = 0.f;

  m[0][1] = 0.f;
  m[1][1] = 2.f / (top - bottom);
  m[2][1] = 0.f;
  m[3][1] = 0.f;

  m[0][2] = 0.f;
  m[1][2] = 0.f;
  m[2][2] = 2.f / (far - near);
  m[3][2] = 0.f;

  m[0][3] = -(right + left) / (right - left);
  m[1][3] = -(top + bottom) / (top - bottom);
  m[2][3] = -(far + near) / (far - near);
  m[3][3] = 1.f;

  Matrix4x4<T> projection(m);
  return Transform<T>(projection, inverse(projection));
}

template <typename T>
__host__ __device__ Transform<T> orthographic(T znear, T zfar) {
  return scale(1.f, 1.f, 1.f / (zfar - znear)) *
         translate(Vector3<T>(0.f, 0.f, -znear));
}
