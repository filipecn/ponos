#ifndef POSEIDON_FAST_SWEEP_H
#define POSEIDON_FAST_SWEEP_H

#include <ponos.h>

namespace poseidon {

inline void solveDistance(float p, float q, float &r) {
  float d = std::min(p, q) + 1;
  if (d > std::max(p, q))
    d = (p + q + sqrtf(2 - SQR(p - q))) / 2;
  if (d < r)
    r = d;
}

template <typename GridType>
void fastSweep2D(GridType *grid, GridType *distances, float largeDist) {
  // +1 +1
  for (int j = 1; j < grid->getDimensions()[1]; j++)
    for (int i = 1; i < grid->getDimensions()[0]; i++) {
      if ((*distances)(i, j) >= largeDist)
        solveDistance((*grid)(i - 1, j), (*grid)(i, j - 1), (*grid)(i, j));
    }
  // -1 -1
  for (int j = grid->getDimensions()[1] - 2; j >= 0; j--)
    for (int i = grid->getDimensions()[0] - 2; i >= 0; i--) {
      if ((*distances)(i, j) >= largeDist)
        solveDistance((*grid)(i + 1, j), (*grid)(i, j + 1), (*grid)(i, j));
    }
  // +1 -1
  for (int j = grid->getDimensions()[1] - 2; j >= 0; j--)
    for (int i = 1; i < grid->getDimensions()[0]; i++) {
      if ((*distances)(i, j) >= largeDist)
        solveDistance((*grid)(i - 1, j), (*grid)(i, j + 1), (*grid)(i, j));
    }
  // -1 +1
  for (int j = 1; j < grid->getDimensions()[1]; j++)
    for (int i = grid->getDimensions()[0] - 2; i >= 0; i--) {
      if ((*distances)(i, j) >= largeDist)
        solveDistance((*grid)(i + 1, j), (*grid)(i, j - 1), (*grid)(i, j));
    }
}

template <typename GridType, typename MaskType, typename T>
void sweep_y(GridType *grid, GridType *phi, MaskType *mask, T MASK_VALUE,
             int i0, int i1, int j0, int j1) {
  int di = (i0 < i1) ? 1 : -1, dj = (j0 < j1) ? 1 : -1;
  float dp, dq, alpha;
  for (int j = j0; j != j1; j += dj)
    for (int i = i0; i != i1; i += di)
      if ((*mask)(i, j - 1) == MASK_VALUE && (*mask)(i, j) == MASK_VALUE) {
        dq = dj * ((*phi)(i, j) - (*phi)(i, j - 1));
        if (dq < 0)
          continue; // not useful on this sweep direction
        dp = 0.5 * ((*phi)(i, j - 1) + (*phi)(i, j) - (*phi)(i - di, j - 1) -
                    (*phi)(i - di, j));
        if (dp < 0)
          continue; // not useful on this sweep direction
        if (dp + dq == 0)
          alpha = 0.5;
        else
          alpha = dp / (dp + dq);
        (*grid)(i, j) =
            alpha * (*grid)(i - di, j) + (1 - alpha) * (*grid)(i, j - dj);
      }
}

template <typename GridType, typename MaskType, typename T>
void sweep_x(GridType *grid, GridType *phi, MaskType *mask, T MASK_VALUE,
             int i0, int i1, int j0, int j1) {
  int di = (i0 < i1) ? 1 : -1, dj = (j0 < j1) ? 1 : -1;
  float dp, dq, alpha;
  for (int j = j0; j != j1; j += dj)
    for (int i = i0; i != i1; i += di)
      if ((*mask)(i - 1, j) == MASK_VALUE && (*mask)(i, j) == MASK_VALUE) {
        dq = dj * ((*phi)(i, j) - (*phi)(i - 1, j));
        if (dq < 0)
          continue; // not useful on this sweep direction
        dp = 0.5 * ((*phi)(i - 1, j) + (*phi)(i, j) - (*phi)(i - 1, j - dj) -
                    (*phi)(i, j - dj));
        if (dp < 0)
          continue; // not useful on this sweep direction
        if (dp + dq == 0)
          alpha = 0.5;
        else
          alpha = dp / (dp + dq);
        (*grid)(i, j) =
            alpha * (*grid)(i - di, j) + (1 - alpha) * (*grid)(i, j - dj);
      }
}

} // poseidon namespace

#endif // POSEIDON_FAST_SWEEP_H
