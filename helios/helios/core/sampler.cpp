#include "sampler.h"

namespace helios {

Sampler::Sampler(int64_t samplesPerPixel) : samplesPerPixel(samplesPerPixel) {}

CameraSample Sampler::getCameraSample(const ponos::point2i &pRaster) {
  CameraSample cs;
  cs.pFilm = (ponos::point2f)pRaster + get2D();
  cs.time = get1D();
  cs.pLens = get2D();
  return cs;
}

int Sampler::roundCount(int n) const { return n; }

void Sampler::startPixel(const ponos::point2i &p) {
  currentPixel = p;
  currentPixelSampleIndex = 0;
  // reset array offsets for next pixel sample
  array1DOffset = array2DOffset = 0;
}

bool Sampler::startNextSample() {
  // reset array offsets for next pixel sample
  array1DOffset = array2DOffset = 0;
  return ++currentPixelSampleIndex < samplesPerPixel;
}

bool Sampler::setSampleNumber(int64_t sampleNum) {
  // reset array offsets for next pixel sample
  array1DOffset = array2DOffset = 0;
  currentPixelSampleIndex = sampleNum;
  return currentPixelSampleIndex < samplesPerPixel;
}

void Sampler::request1DArray(int n) {
  samples1DArraySizes.emplace_back(n);
  sampleArray1D.emplace_back(std::vector<real_t>(n * samplesPerPixel));
}

void Sampler::request2DArray(int n) {
  samples2DArraySizes.emplace_back(n);
  sampleArray2D.emplace_back(std::vector<ponos::point2>(n * samplesPerPixel));
}

const real_t *Sampler::get1DArray(int n) {
  if (array1DOffset == sampleArray1D.size())
    return nullptr;
  return &sampleArray1D[array1DOffset++][currentPixelSampleIndex * n];
}

const ponos::point2 *Sampler::get2DArray(int n) {
  if (array2DOffset == sampleArray2D.size())
    return nullptr;
  return &sampleArray2D[array2DOffset++][currentPixelSampleIndex * n];
}

PixelSampler::PixelSampler(int64_t samplesPerPixel, int nSampledDimensions)
    : Sampler(samplesPerPixel) {
  for (int i = 0; i < nSampledDimensions; ++i) {
    samples1D.emplace_back(std::vector<real_t>(samplesPerPixel));
    samples2D.emplace_back(std::vector<ponos::point2>(samplesPerPixel));
  }
}

bool PixelSampler::startNextSample() {
  current1DDimension = current2DDimension = 0;
  return Sampler::startNextSample();
}

bool PixelSampler::setSampleNumber(int64_t sampleNum) {
  current1DDimension = current2DDimension = 0;
  return Sampler::setSampleNumber(sampleNum);
}

real_t PixelSampler::get1D() {
  if (current1DDimension < samples1D.size())
    return samples1D[current1DDimension++][currentPixelSampleIndex];
  return rng.uniformFloat();
}

ponos::point2 PixelSampler::get2D() {
  if (current2DDimension < samples2D.size())
    return samples2D[current2DDimension++][currentPixelSampleIndex];
  return rng.uniformPoint();
}

GlobalSampler::GlobalSampler(int64_t samplesPerPixel)
    : Sampler(samplesPerPixel) {}

void GlobalSampler::startPixel(const ponos::point2i &p) {
  Sampler::startPixel(p);
  dimension = 0;
  intervalSampleIndex = getIndexForSample(0);
  // compute arrayEndDim for dimensions used for array samples
  arrayEndDim = arrayStartDim + sampleArray1D.size() + 2 * sampleArray2D.size();
  // compute 1D array samples for GlobalSampler
  for (size_t i = 0; i < samples1DArraySizes.size(); ++i) {
    int nSamples = samples1DArraySizes[i] * samplesPerPixel;
    for (int j = 0; j < nSamples; ++j) {
      int64_t index = getIndexForSample(j);
      sampleArray1D[i][j] = sampleDimension(index, arrayStartDim + i);
    }
  }
  // compute 2D array samples for GlobalSampler
  int dim = arrayStartDim + samples1DArraySizes.size();
  for (size_t i = 0; i < samples2DArraySizes.size(); ++i) {
    int nSamples = samples2DArraySizes[i] * samplesPerPixel;
    for (int j = 0; j < nSamples; ++j) {
      int64_t idx = getIndexForSample(j);
      sampleArray2D[i][j].x = sampleDimension(idx, dim);
      sampleArray2D[i][j].y = sampleDimension(idx, dim + 1);
    }
    dim += 2;
  }
}

bool GlobalSampler::startNextSample() {
  dimension = 0;
  intervalSampleIndex = getIndexForSample(currentPixelSampleIndex + 1);
  return Sampler::startNextSample();
}

bool GlobalSampler::setSampleNumber(int64_t sampleNum) {
  dimension = 0;
  intervalSampleIndex = getIndexForSample(sampleNum);
  return Sampler::setSampleNumber(sampleNum);
}

real_t GlobalSampler::get1D() {
  if (dimension >= arrayStartDim && dimension < arrayEndDim)
    dimension = arrayEndDim;
  return sampleDimension(intervalSampleIndex, dimension++);
}

ponos::point2 GlobalSampler::get2D() {
  if (dimension + 1 >= arrayStartDim && dimension < arrayEndDim)
    dimension = arrayEndDim;
  ponos::point2 p(sampleDimension(intervalSampleIndex, dimension),
                  sampleDimension(intervalSampleIndex, dimension + 1));
  dimension += 2;
  return p;
}

void Sampler::stratifiedSample1D(real_t *samp, int nSamples, RNG &rng,
                                 bool jitter) {
  realt_t invNSample = static_cast<real_t>(1) / nSamples;
  for (int i = 0; i < nSamples; ++i) {
    real_t delta = jitter ? rng.uniformFloat() : 0.5f;
    samp[i] = std::min((i + delta) * invNSample, oneMinusEpsilon);
  }
}

void Sampler::stratifiedSample2D(ponos::point2 *samp, int nx, int ny, RNG &rng,
                                 bool jitter) {
  realt_t dx = static_cast<real_t>(1) / nx;
  realt_t dy = static_cast<real_t>(1) / ny;
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      real_t jx = jitter ? rng.uniformFloat() : 0.5f;
      real_t jy = jitter ? rng.uniformFloat() : 0.5f;
      samp->x = std::min((x + jx) * dx, oneMinusEpsilon);
      samp->y = std::min((y + jy) * dx, oneMinusEpsilon);
      ++samp;
    }
}

void Sampler::latinHypercube(real_t *samples, int nSamples, int nDim,
                             RNG &rng) {
  // generate LHS samples along diagonal
  real_t invNSamples = static_cast<real_t>(1) / nSamples;
  for (int i = 0; i < nSamples; ++i)
    for (int j = 0; j < nDim; ++j) {
      real_t sj = (i + (rng.uniformFloat())) * invNSamples;
    }
  // permute LHS samples in each dimension
  for (int i = 0; i < nDim; ++i)
    for (int j = 0; j < nSamples; ++j) {
      int other = j + rng.uniformUInt32(nSamples - j);
      std::swap(samples[nDim * j + i], samples[nDim * other + i]);
    }
}

// Sampler::Sampler(int xstart, int xend, int ystart, int yend, int spp,
//                  float sopen, float sclose)
//     : xPixelStart(xstart), xPixelEnd(xend), yPixelStart(ystart),
//       yPixelEnd(yend), samplesPerPixel(spp), shutterOpen(sopen),
//       shutterClose(sclose) {}

// bool Sampler::reportResults(Sample *samples, const RayDifferential *rays,
//                             const Spectrum *Ls, const Intersection *isects,
//                             int count) {
//   return true;
// }

// void Sampler::computeSubWindow(int num, int count, int *newXStart, int
// *newXEnd,
//                                int *newYStart, int *newYEnd) const {
//   // Determine how many tiles to use in each dimension, nx, and ny
//   int dx = xPixelEnd - xPixelStart, dy = yPixelEnd - yPixelStart;
//   int nx = count, ny = 1;
//   while ((nx & 0x1) == 0 && 2 * dx * ny < dy * nx) {
//     nx >>= 1;
//     ny <<= 1;
//   }
//   // Compute x and y pixel sample range for sub-window
//   int xo = num % nx, yo = num / nx;
//   float tx0 = float(xo) / float(nx), tx1 = float(xo + 1) / float(nx);
//   float ty0 = float(yo) / float(ny), ty1 = float(yo + 1) / float(ny);
//   *newXStart = ponos::floor2Int(ponos::lerp(tx0, xPixelStart, xPixelEnd));
//   *newXEnd = ponos::floor2Int(ponos::lerp(tx1, xPixelStart, xPixelEnd));
//   *newYStart = ponos::floor2Int(ponos::lerp(ty0, yPixelStart, yPixelEnd));
//   *newYEnd = ponos::floor2Int(ponos::lerp(ty1, yPixelStart, yPixelEnd));
// }

// Sample::Sample(Sampler *sampler, SurfaceIntegrator *surf, VolumeIntegrator
// *vol,
//                const Scene *scene) {
//   // if(surf) surf->requestSamples(sampler, this, scene);
//   // if(vol) vol->requestSamples(sampler, this, scene);
//   allocateSampleMemory();
// }

// void Sample::allocateSampleMemory() {
//   // Allocate storage for sample pointers
//   int np = n1D.size() + n2D.size();
//   if (!np) {
//     oneD = twoD = nullptr;
//     return;
//   }
//   oneD = ponos::allocAligned<float *>(np);
//   twoD = oneD + n1D.size();
//   // Compute total number of sample values needed
//   int totalSamples = 0;
//   for (uint32 i : n1D)
//     totalSamples += i;
//   for (uint32 i : n2D)
//     totalSamples += i;
//   // Allocate storage for sample values
//   float *mem = ponos::allocAligned<float>(totalSamples);
//   for (uint32 i = 0; i < n1D.size(); i++) {
//     oneD[i] = mem;
//     mem += n1D[i];
//   }
//   for (uint32 i = 0; i < n2D.size(); i++) {
//     twoD[i] = mem;
//     mem += n2D[i];
//   }
// }

// void generateStratifiedSample1D(float *samp, int nSamples, ponos::RNG &rng,
//                                 bool jitter) {
//   float invTot = 1.f / nSamples;
//   for (int i = 0; i < nSamples; i++) {
//     float delta = jitter ? rng.randomFloat() : 0.5f;
//     *samp++ = (i + delta) * invTot;
//   }
// }

// void generateStratifiedSample2D(float *samp, int nx, int ny, ponos::RNG &rng,
//                                 bool jitter) {
//   float dx = 1.f / nx, dy = 1.f / ny;
//   for (int y = 0; y < ny; y++)
//     for (int x = 0; x < nx; x++) {
//       float jx = jitter ? rng.randomFloat() : 0.5f;
//       float jy = jitter ? rng.randomFloat() : 0.5f;
//       *samp++ = (x + jx) * dx;
//       *samp++ = (y + jy) * dy;
//     }
// }

// void generateLatinHypercube(float *samples, uint32 nSamples, uint32 nDim,
//                             ponos::RNG &rng) {
//   // Generate LHS samples along diagonal
//   float delta = 1.f / nSamples;
//   for (uint32 i = 0; i < nSamples; i++)
//     for (uint32 j = 0; j < nDim; j++)
//       samples[nDim * i + j] = (i + (rng.randomFloat())) * delta;
//   // Permute LHS samples in each dimension
//   for (uint32 i = 0; i < nDim; i++)
//     for (uint32 j = 0; j < nSamples; j++) {
//       uint32 other = j + (rng.randomUInt() % (nSamples - j));
//       std::swap(samples[nDim * j + i], samples[nDim * other + i]);
//     }
// }

} // namespace helios
