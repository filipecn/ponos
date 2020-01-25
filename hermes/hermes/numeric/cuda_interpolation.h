/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * iM the Software without restriction, including without limitation the rights
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

#ifndef HERMES_NUMERIC_CUDA_INTERPOLATION_H
#define HERMES_NUMERIC_CUDA_INTERPOLATION_H

#include <hermes/geometry/cuda_numeric.h>
#include <hermes/geometry/cuda_point.h>

namespace hermes {

namespace cuda {

inline __host__ __device__ float monotonicCubicInterpolate(float fkm1, float fk,
                                                           float fkp1,
                                                           float fkp2,
                                                           float tmtk) {
  double Dk = fkp1 - fk;
  double dk = (fkp1 - fkm1) * 0.5f;
  double dkp1 = (fkp2 - fk) * 0.5f;
  if (fabsf(Dk) < 1e-12f)
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
  float ans = a3 * tmtk * tmtk * tmtk + a2 * tmtk * tmtk + a1 * tmtk + a0;
  float m = fminf(fkm1, fminf(fk, fminf(fkp1, fkp2)));
  float M = fmaxf(fkm1, fmaxf(fk, fmaxf(fkp1, fkp2)));
  return fminf(M, fmaxf(m, ans));
}

inline __host__ __device__ float monotonicCubicInterpolate(float f[4][4][4],
                                                           const point3f &gp) {
  point3f t = gp - vec3f((int)gp.x, (int)gp.y, (int)gp.z);
  float v[4][4];
  for (int dz = -1, iz = 0; dz <= 2; dz++, iz++)
    for (int dy = -1, iy = 0; dy <= 2; dy++, iy++)
      v[iy][iz] = monotonicCubicInterpolate(
          f[dz + 1][dy + 1][0], f[dz + 1][dy + 1][1], f[dz + 1][dy + 1][2],
          f[dz + 1][dy + 1][3], t.x);
  float vv[4];
  for (int d = 0; d < 4; d++)
    vv[d] = monotonicCubicInterpolate(v[d][0], v[d][1], v[d][2], v[d][3], t.z);
  return monotonicCubicInterpolate(vv[0], vv[1], vv[2], vv[3], t.y);
}

inline __host__ __device__ float monotonicCubicInterpolate(float f[4][4],
                                                           const point2f &gp) {
  point2f t = gp - vec2f((int)gp.x, (int)gp.y);
  float v[4];
  for (int d = 0; d < 4; d++)
    v[d] = monotonicCubicInterpolate(f[d][0], f[d][1], f[d][2], f[d][3], t.x);
  return monotonicCubicInterpolate(v[0], v[1], v[2], v[3], t.y);
}

} // namespace cuda

} // namespace hermes

#endif