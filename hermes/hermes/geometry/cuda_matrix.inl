
template <typename T>
__host__ __device__ Matrix4x4<T>::Matrix4x4(bool isIdentity) {
  memset(m, 0, sizeof(m));
  if (isIdentity)
    for (int i = 0; i < 4; i++)
      m[i][i] = 1.f;
}

template <typename T>
__host__ __device__ Matrix4x4<T>::Matrix4x4(const T mat[16], bool columnMajor) {
  size_t k = 0;
  if (columnMajor)
    for (int c = 0; c < 4; c++)
      for (int l = 0; l < 4; l++)
        m[l][c] = mat[k++];
  else
    for (int l = 0; l < 4; l++)
      for (int c = 0; c < 4; c++)
        m[l][c] = mat[k++];
}

template <typename T> __host__ __device__ Matrix4x4<T>::Matrix4x4(T mat[4][4]) {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      m[i][j] = mat[i][j];
}

template <typename T>
__host__ __device__ Matrix4x4<T>::Matrix4x4(T m00, T m01, T m02, T m03, T m10,
                                            T m11, T m12, T m13, T m20, T m21,
                                            T m22, T m23, T m30, T m31, T m32,
                                            T m33) {
  m[0][0] = m00;
  m[0][1] = m01;
  m[0][2] = m02;
  m[0][3] = m03;
  m[1][0] = m10;
  m[1][1] = m11;
  m[1][2] = m12;
  m[1][3] = m13;
  m[2][0] = m20;
  m[2][1] = m21;
  m[2][2] = m22;
  m[2][3] = m23;
  m[3][0] = m30;
  m[3][1] = m31;
  m[3][2] = m32;
  m[3][3] = m33;
}

template <typename T> __host__ __device__ void Matrix4x4<T>::setIdentity() {
  memset(m, 0, sizeof(m));
  for (int i = 0; i < 4; i++)
    m[i][i] = 1.f;
}

template <typename T>
__host__ __device__ void Matrix4x4<T>::row_major(T *a) const {
  int k = 0;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      a[k++] = m[i][j];
}

template <typename T>
__host__ __device__ void Matrix4x4<T>::column_major(T *a) const {
  int k = 0;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      a[k++] = m[j][i];
}

template <typename T>
__host__ __device__ Matrix4x4<T> Matrix4x4<T>::operator*(const Matrix4x4 &mat) {
  Matrix4x4 r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      r.m[i][j] = m[i][0] * mat.m[0][j] + m[i][1] * mat.m[1][j] +
                  m[i][2] * mat.m[2][j] + m[i][3] * mat.m[3][j];
  return r;
}

template <typename T>
__host__ __device__ bool Matrix4x4<T>::operator==(const Matrix4x4 &_m) const {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      if (!IS_EQUAL(m[i][j], _m.m[i][j]))
        return false;
  return true;
}

template <typename T>
__host__ __device__ bool Matrix4x4<T>::operator!=(const Matrix4x4 &_m) const {
  return !(*this == _m);
}

template <typename T> __host__ __device__ bool Matrix4x4<T>::isIdentity() {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      if (i != j && !IS_EQUAL(m[i][j], 0.f))
        return false;
      else if (i == j && !IS_EQUAL(m[i][j], 1.f))
        return false;
  return true;
}

template <typename T>
__host__ __device__ Matrix4x4<T> Matrix4x4<T>::mul(const Matrix4x4 &m1,
                                                   const Matrix4x4 &m2) {
  Matrix4x4 r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
                  m1.m[i][2] * m2.m[2][j] + m1.m[i][3] * m2.m[3][j];
  return r;
}

template <typename T>
__host__ __device__ std::ostream &operator<<(std::ostream &os,
                                             const Matrix4x4<T> &m) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
      os << m.m[i][j] << " ";
    os << std::endl;
  }
  return os;
}

template <typename T>
__host__ __device__ Matrix4x4<T> rowReduce(const Matrix4x4<T> &p,
                                           const Matrix4x4<T> &q) {
  Matrix4x4<T> l = p, r = q;
  // TODO implement with gauss jordan elimination
  return r;
}

template <typename T>
__host__ __device__ Matrix4x4<T> transpose(const Matrix4x4<T> &m) {
  return Matrix4x4<T>(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0], m.m[0][1],
                      m.m[1][1], m.m[2][1], m.m[3][1], m.m[0][2], m.m[1][2],
                      m.m[2][2], m.m[3][2], m.m[0][3], m.m[1][3], m.m[2][3],
                      m.m[3][3]);
}

// function extracted from MESA implementation of the GLU library
template <typename T>
__host__ __device__ bool gluInvertMatrix(const T m[16], T invOut[16]) {
  T inv[16], det;
  int i;

  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
           m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
           m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
           m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
            m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
           m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
           m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
           m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
            m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
           m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
           m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
            m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
            m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
           m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
           m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det == 0)
    return false;

  det = 1.0 / det;

  for (i = 0; i < 16; i++)
    invOut[i] = inv[i] * det;

  return true;
}
template <typename T>
__host__ __device__

    Matrix4x4<T>
    inverse(const Matrix4x4<T> &m) {
  Matrix4x4<T> r;
  T mm[16], inv[16];
  m.row_major(mm);
  if (gluInvertMatrix(mm, inv)) {
    int k = 0;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        r.m[i][j] = inv[k++];
    return r;
  }

  T det = m.m[0][0] * m.m[1][1] * m.m[2][2] * m.m[3][3] +
          m.m[1][2] * m.m[2][3] * m.m[3][1] * m.m[1][3] +
          m.m[2][1] * m.m[3][2] * m.m[1][1] * m.m[2][3] +
          m.m[3][2] * m.m[1][2] * m.m[2][1] * m.m[3][3] +
          m.m[1][3] * m.m[2][2] * m.m[3][1] * m.m[0][1] +
          m.m[0][1] * m.m[2][3] * m.m[3][2] * m.m[0][2] +
          m.m[2][1] * m.m[3][3] * m.m[0][3] * m.m[2][2] +
          m.m[3][1] * m.m[0][1] * m.m[2][2] * m.m[3][3] +
          m.m[0][2] * m.m[2][3] * m.m[3][1] * m.m[0][3] +
          m.m[2][1] * m.m[3][2] * m.m[0][2] * m.m[0][1] +
          m.m[1][2] * m.m[3][3] * m.m[0][2] * m.m[1][3] +
          m.m[3][1] * m.m[0][3] * m.m[1][1] * m.m[3][2] -
          m.m[0][1] * m.m[1][3] * m.m[3][2] * m.m[0][2] -
          m.m[1][1] * m.m[3][3] * m.m[0][3] * m.m[1][2] -
          m.m[3][1] * m.m[0][3] * m.m[0][1] * m.m[1][3] -
          m.m[2][2] * m.m[0][2] * m.m[1][1] * m.m[2][3] -
          m.m[0][3] * m.m[1][2] * m.m[2][1] * m.m[0][1] -
          m.m[1][2] * m.m[2][3] * m.m[0][2] * m.m[1][3] -
          m.m[2][1] * m.m[0][3] * m.m[1][1] * m.m[2][2] -
          m.m[1][0] * m.m[1][0] * m.m[2][3] * m.m[3][2] -
          m.m[1][2] * m.m[2][0] * m.m[3][3] * m.m[1][3] -
          m.m[2][2] * m.m[3][0] * m.m[1][0] * m.m[2][2] -
          m.m[3][3] * m.m[1][2] * m.m[2][3] * m.m[3][0] -
          m.m[1][3] * m.m[2][0] * m.m[3][2] * m.m[1][1];
  if (fabs(det) < 1e-8)
    return r;

  r.m[0][0] =
      (m.m[1][1] * m.m[2][2] * m.m[3][3] + m.m[1][2] * m.m[2][3] * m.m[3][1] +
       m.m[1][3] * m.m[2][1] * m.m[3][2] - m.m[1][1] * m.m[2][3] * m.m[3][2] -
       m.m[1][2] * m.m[2][1] * m.m[3][3] - m.m[1][3] * m.m[2][2] * m.m[3][1]) /
      det;
  r.m[0][1] =
      (m.m[0][1] * m.m[2][3] * m.m[3][2] + m.m[0][2] * m.m[2][1] * m.m[3][3] +
       m.m[0][3] * m.m[2][2] * m.m[3][1] - m.m[0][1] * m.m[2][2] * m.m[3][3] -
       m.m[0][2] * m.m[2][3] * m.m[3][1] - m.m[0][3] * m.m[2][1] * m.m[3][2]) /
      det;
  r.m[0][2] =
      (m.m[0][1] * m.m[1][2] * m.m[3][3] + m.m[0][2] * m.m[1][3] * m.m[3][1] +
       m.m[0][3] * m.m[1][1] * m.m[3][2] - m.m[0][1] * m.m[1][3] * m.m[3][2] -
       m.m[0][2] * m.m[1][1] * m.m[3][3] - m.m[0][3] * m.m[1][2] * m.m[3][1]) /
      det;
  r.m[0][3] =
      (m.m[0][1] * m.m[1][3] * m.m[2][2] + m.m[0][2] * m.m[1][1] * m.m[2][3] +
       m.m[0][3] * m.m[1][2] * m.m[2][1] - m.m[0][1] * m.m[1][2] * m.m[2][3] -
       m.m[0][2] * m.m[1][3] * m.m[2][1] - m.m[0][3] * m.m[1][1] * m.m[2][2]) /
      det;
  r.m[1][0] =
      (m.m[1][0] * m.m[2][3] * m.m[3][2] + m.m[1][2] * m.m[2][0] * m.m[3][3] +
       m.m[1][3] * m.m[2][2] * m.m[3][0] - m.m[1][0] * m.m[2][2] * m.m[3][3] -
       m.m[1][2] * m.m[2][3] * m.m[3][0] - m.m[1][3] * m.m[2][0] * m.m[3][2]) /
      det;
  r.m[1][1] =
      (m.m[0][0] * m.m[2][2] * m.m[3][3] + m.m[0][2] * m.m[2][3] * m.m[3][0] +
       m.m[0][3] * m.m[2][0] * m.m[3][2] - m.m[0][0] * m.m[2][3] * m.m[3][2] -
       m.m[0][2] * m.m[2][0] * m.m[3][3] - m.m[0][3] * m.m[2][2] * m.m[3][0]) /
      det;
  r.m[1][2] =
      (m.m[0][0] * m.m[1][3] * m.m[3][2] + m.m[0][2] * m.m[1][0] * m.m[3][3] +
       m.m[0][3] * m.m[1][2] * m.m[3][0] - m.m[0][0] * m.m[1][2] * m.m[3][3] -
       m.m[0][2] * m.m[1][3] * m.m[3][0] - m.m[0][3] * m.m[1][0] * m.m[3][2]) /
      det;
  r.m[1][3] =
      (m.m[0][0] * m.m[1][2] * m.m[2][3] + m.m[0][2] * m.m[1][3] * m.m[2][0] +
       m.m[0][3] * m.m[1][0] * m.m[2][2] - m.m[0][0] * m.m[1][3] * m.m[2][2] -
       m.m[0][2] * m.m[1][0] * m.m[2][3] - m.m[0][3] * m.m[1][2] * m.m[2][0]) /
      det;
  r.m[2][0] =
      (m.m[1][0] * m.m[2][1] * m.m[3][3] + m.m[1][1] * m.m[2][3] * m.m[3][0] +
       m.m[1][3] * m.m[2][0] * m.m[3][1] - m.m[1][0] * m.m[2][3] * m.m[3][1] -
       m.m[1][1] * m.m[2][0] * m.m[3][3] - m.m[1][3] * m.m[2][1] * m.m[3][0]) /
      det;
  r.m[2][1] =
      (m.m[0][0] * m.m[2][3] * m.m[3][1] + m.m[0][1] * m.m[2][0] * m.m[3][3] +
       m.m[0][3] * m.m[2][1] * m.m[3][0] - m.m[0][0] * m.m[2][1] * m.m[3][3] -
       m.m[0][1] * m.m[2][3] * m.m[3][0] - m.m[0][3] * m.m[2][0] * m.m[3][1]) /
      det;
  r.m[2][2] =
      (m.m[0][0] * m.m[1][1] * m.m[3][3] + m.m[0][1] * m.m[1][3] * m.m[3][0] +
       m.m[0][3] * m.m[1][0] * m.m[3][1] - m.m[0][0] * m.m[1][3] * m.m[3][1] -
       m.m[0][1] * m.m[1][0] * m.m[3][3] - m.m[0][3] * m.m[1][1] * m.m[3][0]) /
      det;
  r.m[2][3] =
      (m.m[0][0] * m.m[1][3] * m.m[2][1] + m.m[0][1] * m.m[1][0] * m.m[2][3] +
       m.m[0][3] * m.m[1][1] * m.m[2][0] - m.m[0][0] * m.m[1][1] * m.m[2][3] -
       m.m[0][1] * m.m[1][3] * m.m[2][0] - m.m[0][3] * m.m[1][0] * m.m[2][1]) /
      det;
  r.m[3][0] =
      (m.m[1][0] * m.m[2][2] * m.m[3][1] + m.m[1][1] * m.m[2][0] * m.m[3][2] +
       m.m[1][2] * m.m[2][1] * m.m[3][0] - m.m[1][0] * m.m[2][1] * m.m[3][2] -
       m.m[1][1] * m.m[2][2] * m.m[3][0] - m.m[1][2] * m.m[2][0] * m.m[3][1]) /
      det;
  r.m[3][1] =
      (m.m[0][0] * m.m[2][1] * m.m[3][2] + m.m[0][1] * m.m[2][2] * m.m[3][0] +
       m.m[0][2] * m.m[2][0] * m.m[3][1] - m.m[0][0] * m.m[2][2] * m.m[3][1] -
       m.m[0][1] * m.m[2][0] * m.m[3][2] - m.m[0][2] * m.m[2][1] * m.m[3][0]) /
      det;
  r.m[3][2] =
      (m.m[0][0] * m.m[1][2] * m.m[3][1] + m.m[0][1] * m.m[1][0] * m.m[3][2] +
       m.m[0][2] * m.m[1][1] * m.m[3][0] - m.m[0][0] * m.m[1][1] * m.m[3][2] -
       m.m[0][1] * m.m[1][2] * m.m[3][0] - m.m[0][2] * m.m[1][0] * m.m[3][1]) /
      det;
  r.m[3][3] =
      (m.m[0][0] * m.m[1][1] * m.m[2][2] + m.m[0][1] * m.m[1][2] * m.m[2][0] +
       m.m[0][2] * m.m[1][0] * m.m[2][1] - m.m[0][0] * m.m[1][2] * m.m[2][1] -
       m.m[0][1] * m.m[1][0] * m.m[2][2] - m.m[0][2] * m.m[1][1] * m.m[2][0]) /
      det;

  return r;
}

template <typename T>
__host__ __device__ void decompose(const Matrix4x4<T> &m, Matrix4x4<T> &r,
                                   Matrix4x4<T> &s) {
  // extract rotation r from transformation matrix
  T norm;
  int count = 0;
  r = m;
  do {
    // compute next matrix in series
    Matrix4x4<T> Rnext;
    Matrix4x4<T> Rit = inverse(transpose(r));
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        Rnext.m[i][j] = .5f * (r.m[i][j] + Rit.m[i][j]);
    // compute norm difference between R and Rnext
    norm = 0.f;
    for (int i = 0; i < 3; i++) {
      T n = fabsf(r.m[i][0] - Rnext.m[i][0]) +
            fabsf(r.m[i][1] - Rnext.m[i][1]) + fabsf(r.m[i][2] - Rnext.m[i][2]);
      norm = fmaxf(norm, n);
    }
  } while (++count < 100 && norm > .0001f);
  // compute scale S using rotation and original matrix
  s = Matrix4x4<T>::mul(inverse(r), m);
}

template <typename T> __host__ __device__ Matrix3x3<T>::Matrix3x3() {
  memset(m, 0, sizeof(m));
}

template <typename T>
__host__ __device__ Matrix3x3<T>::Matrix3x3(vec3 a, vec3 b, vec3 c)
    : Matrix3x3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z) {}

template <typename T>
__host__ __device__ Matrix3x3<T>::Matrix3x3(T m00, T m01, T m02, T m10, T m11,
                                            T m12, T m20, T m21, T m22) {
  m[0][0] = m00;
  m[0][1] = m01;
  m[0][2] = m02;
  m[1][0] = m10;
  m[1][1] = m11;
  m[1][2] = m12;
  m[2][0] = m20;
  m[2][1] = m21;
  m[2][2] = m22;
}

template <typename T> __host__ __device__ void Matrix3x3<T>::setIdentity() {
  memset(m, 0, sizeof(m));
  for (int i = 0; i < 3; i++)
    m[i][i] = 1.f;
}

template <typename T>
__host__ __device__ Vector3<T> Matrix3x3<T>::
operator*(const Vector3<T> &v) const {
  Vector3<T> r;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      r[i] += m[i][j] * v[j];
  return r;
}

template <typename T>
__host__ __device__ Matrix3x3<T> Matrix3x3<T>::operator*(const Matrix3x3 &mat) {
  Matrix3x3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r.m[i][j] =
          m[i][0] * mat.m[0][j] + m[i][1] * mat.m[1][j] + m[i][2] * mat.m[2][j];
  return r;
}

template <typename T>
__host__ __device__ Matrix3x3<T> Matrix3x3<T>::operator*(const T &f) const {
  return Matrix3x3(m[0][0] * f, m[0][1] * f, m[0][2] * f, m[1][0] * f,
                   m[1][1] * f, m[1][2] * f, m[2][0] * f, m[2][1] * f,
                   m[2][2] * f);
}

template <typename T>
__host__ __device__ Matrix3x3<T> Matrix3x3<T>::mul(const Matrix3x3 &m1,
                                                   const Matrix3x3 &m2) {
  Matrix3x3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
                  m1.m[i][2] * m2.m[2][j];
  return r;
}

template <typename T> __host__ __device__ T Matrix3x3<T>::determinant() {
  return m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] +
         m[0][2] * m[1][0] * m[2][1] - m[2][0] * m[1][1] * m[0][2] -
         m[2][1] * m[1][2] * m[0][0] - m[2][2] * m[1][0] * m[0][1];
}

template <typename T>
__host__ __device__ Matrix3x3<T> inverse(const Matrix3x3<T> &m) {
  Matrix3x3<T> r;
  // r.setIdentity();
  T det =
      m.m[0][0] * m.m[1][1] * m.m[2][2] + m.m[1][0] * m.m[2][1] * m.m[0][2] +
      m.m[2][0] * m.m[0][1] * m.m[1][2] - m.m[0][0] * m.m[2][1] * m.m[1][2] -
      m.m[2][0] * m.m[1][1] * m.m[0][2] - m.m[1][0] * m.m[0][1] * m.m[2][2];
  if (fabs(det) < 1e-8)
    return r;
  r.m[0][0] = (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) / det;
  r.m[0][1] = (m.m[0][2] * m.m[2][1] - m.m[0][1] * m.m[2][2]) / det;
  r.m[0][2] = (m.m[0][1] * m.m[1][2] - m.m[0][2] * m.m[1][1]) / det;
  r.m[1][0] = (m.m[1][2] * m.m[2][0] - m.m[1][0] * m.m[2][2]) / det;
  r.m[1][1] = (m.m[0][0] * m.m[2][2] - m.m[0][2] * m.m[2][0]) / det;
  r.m[1][2] = (m.m[0][2] * m.m[1][0] - m.m[0][0] * m.m[1][2]) / det;
  r.m[2][0] = (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]) / det;
  r.m[2][1] = (m.m[0][1] * m.m[2][0] - m.m[0][0] * m.m[2][1]) / det;
  r.m[2][2] = (m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0]) / det;
  return r;
}

template <typename T>
__host__ __device__ Matrix3x3<T> transpose(const Matrix3x3<T> &m) {
  return Matrix3x3<T>(m.m[0][0], m.m[1][0], m.m[2][0], m.m[0][1], m.m[1][1],
                      m.m[2][1], m.m[0][2], m.m[1][2], m.m[2][2]);
}

template <typename T>
__host__ __device__ Matrix3x3<T> star(const Vector3<T> a) {
  return Matrix3x3<T>(0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0);
}

template <typename T>
__host__ __device__ std::ostream &operator<<(std::ostream &os,
                                             const Matrix3x3<T> &m) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      os << m.m[i][j] << " ";
    os << std::endl;
  }
  return os;
}
