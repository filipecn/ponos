#ifndef PONOS_GEOMETRY_INTERPOLATION_H
#define PONOS_GEOMETRY_INTERPOLATION_H

#include <algorithm>
#include <ponos/common/size.h>
#include <ponos/numeric/numeric.h>

namespace ponos {
/// \param t **[in]** (in [0,1]) parametric coordinate of the interpolation
/// point
/// \param a **[in]** lower bound **0**
/// \param b **[in]** upper bound **1**
/// \return linear interpolation between **a** and **b** at **t**.
template <typename T> T lerp(T t, T a, T b) { return (1. - t) * a + t * b; }
/// \param x **[in]** (in [0,1]) parametric coordinate in x
/// \param y **[in]** (in [0,1]) parametric coordinate in y
/// \param f00 **[in]** function value at **(0, 0)**
/// \param f10 **[in]** function value at **(1, 0)**
/// \param f11 **[in]** function value at **(1, 1)**
/// \param f01 **[in]** function value at **(0, 1)**
/// \return bilinear interpolation value at **(x, y)**
template <typename T>
inline T bilerp(T x, T y, const T &f00, const T &f10, const T &f11,
                const T &f01) {
  return lerp(y, lerp(x, f00, f10), lerp(x, f01, f11));
}
/// \tparam T
/// \tparam S
/// \param tx **[in]** (in [0,1]) parametric coordinate in x
/// \param ty **[in]** (in [0,1]) parametric coordinate in y
/// \param tz **[in]** (in [0,1]) parametric coordinate in z
/// \param f000 **[in]** function value at **(0, 0, 0)**
/// \param f100 **[in]** function value at **(1, 0, 0)**
/// \param f010 **[in]** function value at **(0, 1, 0)**
/// \param f110 **[in]** function value at **(1, 1, 0)**
/// \param f001 **[in]** function value at **(0, 0, 1)**
/// \param f101 **[in]** function value at **(1, 0, 1)**
/// \param f011 **[in]** function value at **(0, 1, 1)**
/// \param f111 **[in]** function value at **(1, 1, 1)**
/// \return trilinear interpolation value at **(x, y, z)**
template <typename T>
inline T trilerp(T tx, T ty, T tz, const T &f000, const T &f100, const T &f010,
                 const T &f110, const T &f001, const T &f101, const T &f011,
                 const T &f111) {
  return lerp(bilerp(f000, f100, f010, f110, tx, ty),
              bilerp(f001, f101, f011, f111, tx, ty), tz);
}
/** \brief interpolates to the nearest value
 * \param t **[in]** parametric coordinate
 * \param a **[in]** lower bound
 * \param b **[in]** upper bound
 * \return interpolation between **a** and **b** at **t**.
 */
template <typename T> inline T nearest(T t, const T &a, const T &b) {
  return (t < static_cast<T>(0.5)) ? a : b;
}
/// Performs 1-dimensional interpolation using the monotonic cubic
/// interpolation considering overshoot, clamps the resulting value.
/// Assumes the following configuration:
/// fkm1         fk           fkp1         fkp2
/// * ---------- * ----x----- * ---------- *
/// -1           0     tmtk   1            2
/// \tparam T data type
/// \param fkm1 function value at -1
/// \param fk function value at 0
/// \param fkp1 function value at 1
/// \param fkp2 function value at 2
/// \param tmtk (in [0,1]) parametric coordinate of the interpolation
/// point
/// \return interpolated value at tmtk
template <typename T>
T monotonicCubicInterpolate(T fkm1, T fk, T fkp1, T fkp2, T tmtk) {
  double Dk = fkp1 - fk;
  double dk = (fkp1 - fkm1) * 0.5;
  double dkp1 = (fkp2 - fk) * 0.5;
  if (fabsf(Dk) < 1e-12)
    dk = dkp1 = 0.0;
  else {
    if (sign(dk) != sign(Dk))
      // dk = 0;
      dk *= -1;
    if (sign(dkp1) != sign(Dk))
      // dkp1 = 0;
      dkp1 *= -1;
  }
  double a0 = fk;
  double a1 = dk;
  double a2 = 3 * Dk - 2 * dk - dkp1;
  double a3 = dk + dkp1 - 2 * Dk;
  T ans = a3 * tmtk * tmtk * tmtk + a2 * tmtk * tmtk + a1 * tmtk + a0;
  T m = std::min(fkm1, std::min(fk, std::min(fkp1, fkp2)));
  T M = std::max(fkm1, std::max(fk, std::max(fkp1, fkp2)));
  return std::min(M, std::max(m, ans));
}
/// Performs a 2-dimensional interpolation using the monotonic cubic
/// interpolation considering overshoot, clamps the resulting value.
/// Assumes the sampling point at the cell between indices 1 and 2.
/// (1,1) <= x <= (2,2)
///    f[0][3] --- f[1][3] --- f[2][3] --- f[3][3]
///    |           |           |           |
///    f[0][2] --- f[1][2] --- f[2][2] --- f[3][2]
///    |           |      x    |           |
///    f[0][1] --- f[1][1] --- f[2][1] --- f[3][1]
///    |           |           |           |
///    f[0][0] --- f[1][0] --- f[2][0] --- f[3][0]
/// \tparam T data type
/// \param f function values at grid points (f[x][y])
/// \param cp (in [0,1]) cell parametric coordinates
/// \return interpolated value at cp
template <typename T>
float monotonicCubicInterpolate(T f[4][4], const point2 &t) {
  T v[4];
  for (int d = 0; d < 4; d++)
    v[d] = monotonicCubicInterpolate(f[0][d], f[1][d], f[2][d], f[3][d], t.x);
  return monotonicCubicInterpolate(v[0], v[1], v[2], v[3], t.y);
}
/// Performs a 3-dimensional interpolation using the monotonic cubic
/// interpolation considering overshoot, clamps the resulting value.
/// Assumes the sampling point x at the cube between indices 1 and 2:
/// (1,1,1) <= x <= (2,2,2)
/// \tparam T data type
/// \param f function values at grid points (f[x][y][z])
/// \param cp (in [0,1]) cell parametric coordinates
/// \return interpolated value at cp
template <typename T>
float monotonicCubicInterpolate(T f[4][4][4], const point3 &t) {
  float v[4][4];
  for (int dz = -1, iz = 0; dz <= 2; dz++, iz++)
    for (int dy = -1, iy = 0; dy <= 2; dy++, iy++)
      v[iy][iz] = monotonicCubicInterpolate(
          f[0][dy + 1][dz + 1], f[1][dy + 1][dz + 1], f[2][dy + 1][dz + 1],
          f[3][dy + 1][dz + 1], t.x);
  float vv[4];
  for (int d = 0; d < 4; d++)
    vv[d] = monotonicCubicInterpolate(v[0][d], v[1][d], v[2][d], v[3][d], t.y);
  return monotonicCubicInterpolate(vv[0], vv[1], vv[2], vv[3], t.z);
}
template <typename S, typename T>
inline S catmullRomSpline(const S &f0, const S &f1, const S &f2, const S &f3,
                          T f) {
  S d1 = (f2 - f0) / 2;
  S d2 = (f3 - f1) / 2;
  S D1 = f2 - f1;

  S a3 = d1 + d2 - 2 * D1;
  S a2 = 3 * D1 - 2 * d1 - d2;
  S a1 = d1;
  S a0 = f1;

  return a3 * CUBE(f) + a2 * SQR(f) + a1 * f + a0;
}
template <typename T>
T bilinearInterpolation(T f00, T f10, T f11, T f01, T x, T y) {
  return f00 * (1.0 - x) * (1.0 - y) + f10 * x * (1.0 - y) +
         f01 * (1.0 - x) * y + f11 * x * y;
}
template <typename T> inline T cubicInterpolate(T p[4], T x) {
  return p[1] + 0.5 * x *
                    (p[2] - p[0] +
                     x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] +
                          x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}
template <typename T> inline T bicubicInterpolate(T p[4][4], T x, T y) {
  T arr[4];
  arr[0] = cubicInterpolate(p[0], y);
  arr[1] = cubicInterpolate(p[1], y);
  arr[2] = cubicInterpolate(p[2], y);
  arr[3] = cubicInterpolate(p[3], y);
  return cubicInterpolate(arr, x);
}
template <typename T>
inline T trilinearInterpolate(float *p, T ***data, T b,
                              const size3 &dimensions) {
  i32 i0 = p[0], j0 = p[1], k0 = p[2];
  i32 i1 = p[0] + 1, j1 = p[1] + 1, k1 = p[2] + 1;
  float x = p[0] - i0;
  float y = p[1] - j0;
  float z = p[2] - k0;
  T v000 = (i0 < 0 || j0 < 0 || k0 < 0 || i0 >= dimensions[0] ||
            j0 >= dimensions[1] || k0 >= dimensions[2])
               ? b
               : data[i0][j0][k0];
  T v001 = (i0 < 0 || j0 < 0 || k1 < 0 || i0 >= dimensions[0] ||
            j0 >= dimensions[1] || k1 >= dimensions[2])
               ? b
               : data[i0][j0][k1];
  T v010 = (i0 < 0 || j1 < 0 || k0 < 0 || i0 >= dimensions[0] ||
            j1 >= dimensions[1] || k0 >= dimensions[2])
               ? b
               : data[i0][j1][k0];
  T v011 = (i0 < 0 || j1 < 0 || k1 < 0 || i0 >= dimensions[0] ||
            j1 >= dimensions[1] || k1 >= dimensions[2])
               ? b
               : data[i0][j1][k1];
  T v100 = (i1 < 0 || j0 < 0 || k0 < 0 || i1 >= dimensions[0] ||
            j0 >= dimensions[1] || k0 >= dimensions[2])
               ? b
               : data[i1][j0][k0];
  T v101 = (i1 < 0 || j0 < 0 || k1 < 0 || i1 >= dimensions[0] ||
            j0 >= dimensions[1] || k1 >= dimensions[2])
               ? b
               : data[i1][j0][k1];
  T v110 = (i1 < 0 || j1 < 0 || k0 < 0 || i1 >= dimensions[0] ||
            j1 >= dimensions[1] || k0 >= dimensions[2])
               ? b
               : data[i1][j1][k0];
  T v111 = (i1 < 0 || j1 < 0 || k1 < 0 || i1 >= dimensions[0] ||
            j1 >= dimensions[1] || k1 >= dimensions[2])
               ? b
               : data[i1][j1][k1];
  return v000 * (1.f - x) * (1.f - y) * (1.f - z) +
         v100 * x * (1.f - y) * (1.f - z) + v010 * (1.f - x) * y * (1.f - z) +
         v110 * x * y * (1.f - z) + v001 * (1.f - x) * (1.f - y) * z +
         v101 * x * (1.f - y) * z + v011 * (1.f - x) * y * z + v111 * x * y * z;
}
template <typename T> inline T tricubicInterpolate(float *p, T ***data) {
  int x, y, z;
  int i, j, k;
  float dx, dy, dz;
  float u[4], v[4], w[4];
  T r[4], q[4];
  T vox = T(0);

  x = (int)p[0], y = (int)p[1], z = (int)p[2];
  dx = p[0] - (float)x, dy = p[1] - (float)y, dz = p[2] - (float)z;

  u[0] = -0.5 * CUBE(dx) + SQR(dx) - 0.5 * dx;
  u[1] = 1.5 * CUBE(dx) - 2.5 * SQR(dx) + 1;
  u[2] = -1.5 * CUBE(dx) + 2 * SQR(dx) + 0.5 * dx;
  u[3] = 0.5 * CUBE(dx) - 0.5 * SQR(dx);

  v[0] = -0.5 * CUBE(dy) + SQR(dy) - 0.5 * dy;
  v[1] = 1.5 * CUBE(dy) - 2.5 * SQR(dy) + 1;
  v[2] = -1.5 * CUBE(dy) + 2 * SQR(dy) + 0.5 * dy;
  v[3] = 0.5 * CUBE(dy) - 0.5 * SQR(dy);

  w[0] = -0.5 * CUBE(dz) + SQR(dz) - 0.5 * dz;
  w[1] = 1.5 * CUBE(dz) - 2.5 * SQR(dz) + 1;
  w[2] = -1.5 * CUBE(dz) + 2 * SQR(dz) + 0.5 * dz;
  w[3] = 0.5 * CUBE(dz) - 0.5 * SQR(dz);

  int ijk[3] = {x - 1, y - 1, z - 1};
  for (k = 0; k < 4; k++) {
    q[k] = 0;
    for (j = 0; j < 4; j++) {
      r[j] = 0;
      for (i = 0; i < 4; i++) {
        r[j] += u[i] * data[ijk[0]][ijk[1]][ijk[2]];
        ijk[0]++;
      }
      q[k] += v[j] * r[j];
      ijk[0] = x - 1;
      ijk[1]++;
    }
    vox += w[k] * q[k];
    ijk[0] = x - 1;
    ijk[1] = y - 1;
    ijk[2]++;
  }
  return (vox < T(0) ? T(0.0) : vox);
}
template <typename T>
T tricubicInterpolate(float *p, T ***data, T b, const int dimensions[3]) {
  int x, y, z;
  int i, j, k;
  float dx, dy, dz;
  float u[4], v[4], w[4];
  T r[4], q[4];
  T vox = T(0);

  x = (int)p[0], y = (int)p[1], z = (int)p[2];
  dx = p[0] - (float)x, dy = p[1] - (float)y, dz = p[2] - (float)z;

  u[0] = -0.5 * CUBE(dx) + SQR(dx) - 0.5 * dx;
  u[1] = 1.5 * CUBE(dx) - 2.5 * SQR(dx) + 1;
  u[2] = -1.5 * CUBE(dx) + 2 * SQR(dx) + 0.5 * dx;
  u[3] = 0.5 * CUBE(dx) - 0.5 * SQR(dx);

  v[0] = -0.5 * CUBE(dy) + SQR(dy) - 0.5 * dy;
  v[1] = 1.5 * CUBE(dy) - 2.5 * SQR(dy) + 1;
  v[2] = -1.5 * CUBE(dy) + 2 * SQR(dy) + 0.5 * dy;
  v[3] = 0.5 * CUBE(dy) - 0.5 * SQR(dy);

  w[0] = -0.5 * CUBE(dz) + SQR(dz) - 0.5 * dz;
  w[1] = 1.5 * CUBE(dz) - 2.5 * SQR(dz) + 1;
  w[2] = -1.5 * CUBE(dz) + 2 * SQR(dz) + 0.5 * dz;
  w[3] = 0.5 * CUBE(dz) - 0.5 * SQR(dz);

  int ijk[3] = {x - 1, y - 1, z - 1};
  for (k = 0; k < 4; k++) {
    q[k] = 0;
    for (j = 0; j < 4; j++) {
      r[j] = 0;
      for (i = 0; i < 4; i++) {
        if (ijk[0] < 0 || ijk[0] >= dimensions[0] || ijk[1] < 0 ||
            ijk[1] >= dimensions[1] || ijk[2] < 0 || ijk[2] >= dimensions[2])
          r[j] += u[i] * b;
        else
          r[j] += u[i] * data[ijk[0]][ijk[1]][ijk[2]];
        ijk[0]++;
      }
      q[k] += v[j] * r[j];
      ijk[0] = x - 1;
      ijk[1]++;
    }
    vox += w[k] * q[k];
    ijk[0] = x - 1;
    ijk[1] = y - 1;
    ijk[2]++;
  }
  return (vox < T(0) ? T(0.0) : vox);
}

} // namespace ponos

#endif