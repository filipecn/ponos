#ifndef PONOS_GEOMETRY_INTERPOLATION_H
#define PONOS_GEOMETRY_INTERPOLATION_H

#include <ponos/common/size.h>

namespace ponos {

template <typename T>
inline T bilinearInterpolation(T f00, T f10, T f11, T f01, T x, T y) {
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
  int i0 = p[0], j0 = p[1], k0 = p[2];
  int i1 = p[0] + 1, j1 = p[1] + 1, k1 = p[2] + 1;
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
inline T tricubicInterpolate(float *p, T ***data, T b,
                             const int dimensions[3]) {
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