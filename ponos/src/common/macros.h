#ifndef PONOS_COMMON_MACROS_H
#define PONOS_COMMON_MACROS_H

#define FOR_LOOP(i, x0, x1) for (i = 0; i < x1; ++i)

#define FOR_INDICES0_2D(D, ij)                                                 \
  for (ij[0] = 0; ij[0] < (D[0]); ++ij[0])                                     \
    for (ij[1] = 0; ij[1] < (D[1]); ++ij[1])

#define FOR_INDICES0_E2D(D, ij)                                                \
  for (ij[0] = 0; ij[0] <= (D[0]); ++ij[0])                                    \
    for (ij[1] = 0; ij[1] <= (D[1]); ++ij[1])

#define FOR_INDICES(x0, x1, y0, y1, z0, z1, ijk)                               \
  for (ijk[0] = (x0); ijk[0] < (x1); ++ijk[0])                                 \
    for (ijk[1] = (y0); ijk[1] < (y1); ++ijk[1])                               \
      for (ijk[2] = (z0); ijk[2] < (z1); ++ijk[2])

#define FOR_INDICES0_3D(D, ijk)                                                \
  for (ijk[0] = 0; ijk[0] < (D[0]); ++ijk[0])                                  \
    for (ijk[1] = 0; ijk[1] < (D[1]); ++ijk[1])                                \
      for (ijk[2] = 0; ijk[2] < (D[2]); ++ijk[2])

#define FOR_INDICES0_3D_ijk(D, i, j, k)                                        \
  for (int i = 0; i < (D[0]); ++i)                                             \
    for (int j = 0; j < (D[1]); ++j)                                           \
      for (int k = 0; k < (D[2]); ++k)

#define FOR_INDICES3D(D0, D1, ijk)                                             \
  for (ijk[0] = D0[0]; ijk[0] < (D1[0]); ++ijk[0])                             \
    for (ijk[1] = D0[1]; ijk[1] < (D1[1]); ++ijk[1])                           \
      for (ijk[2] = D0[2]; ijk[2] < (D1[2]); ++ijk[2])

#define FOR_INDICES2D(D0, D1, ij)                                              \
  for (ij[0] = D0[0]; ij[0] < (D1[0]); ++ij[0])                                \
    for (ij[1] = D0[1]; ij[1] < (D1[1]); ++ij[1])

#endif // PONOS_COMMON_MACROS_H
