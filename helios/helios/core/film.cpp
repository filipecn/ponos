#include <helios/core/film.h>

namespace helios {

FilmTile::FilmTile(const bounds2i &pixelBounds, const ponos::vec2 &filterRadius,
                   const real_t *filterTable, int filterTableSize)
    : pixelBounds(pixelBounds), filterRadius(filterRadius),
      invFilterRadius(1 / filterRadius.x, 1 / filterRadius.y),
      filterTable(filterTable), filterTableSize(filterTableSize) {
  pixels = std::vector<FilmTilePixel>(std::max(0, pixelBounds.area()));
}

void FilmTile::addSample(const ponos::point2 &pFilm, const Spectrum &L,
                         real_t sampleWeight) {
  // compute sample's raster bounds
  ponos::point2 pFilmDiscrete = pFilm - ponos::vec2(0.5f);
  ponos::point2i p0 = (ponos::point2i)ponos::ceil(pFilmDiscrete - filterRadius);
  ponos::point2i p1 =
      (ponos::point2i)ponos::floor(pFilmDiscrete + filterRadius) +
      ponos::vec2i(1);
  p0 = ponos::max(p0, pixelBounds.lower);
  p1 = ponos::min(p1, pixelBounds.upper);
  // loop over filter support and add sample to pixel arrays
  // precompute x and y filter table offsets
  int *ifx = ALLOCA(int, p1.x - p0.x);
  for (int x = p0.x; x < p1.x; ++x) {
    real_t fx =
        std::abs((x - pFilmDiscrete.x) * invFilterRadius.x * filterTableSize);
    ifx[x - p0.x] = std::min((int)std::floor(fx), filterTableSize - 1);
  }
  int *ify = ALLOCA(int, p1.y - p0.y);
  for (int y = p0.y; y < p1.y; ++y) {
    real_t fy =
        std::abs((y - pFilmDiscrete.y) * invFilterRadius.y * filterTableSize);
    ifx[y - p0.y] = std::min((int)std::floor(fy), filterTableSize - 1);
  }
  for (int y = p0.y; y < p1.y; ++y)
    for (int x = p0.x; x < p1.x; ++x) {
      // evaluate filter value at (x, y) pixel
      int offset = ify[y - p0.y] * filterTableSize + ifx[x - p0.x];
      real_t filterWeight = filterTable[offset];
      // update pixel va lues with filtered sample contribution
      FilmTilePixel &pixel = getPixel(ponos::point2i(x, y));
      pixel.contribSum += L * sampleWeight * filterWeight;
      pixel.filterWeightSum += filterWeight;
    }
}

FilmTilePixel &FilmTile::getPixel(const ponos::point2i &p) {
  int width = pixelBounds.upper.x - pixelBounds.lower.x;
  int offset =
      (p.x - pixelBounds.lower.x) + (p.y - pixelBounds.lower.y) * width;
  return pixels[offset];
}

const FilmTilePixel &FilmTile::getPixel(const ponos::point2i &p) const {
  int width = pixelBounds.upper.x - pixelBounds.lower.x;
  int offset =
      (p.x - pixelBounds.lower.x) + (p.y - pixelBounds.lower.y) * width;
  return pixels[offset];
}

bounds2i FilmTile::getPixelBounds() const { return pixelBounds; }

Film::Film(const ponos::point2i &resolution, const bounds2 &cropWindow,
           std::unique_ptr<Filter> filter, real_t diagonal,
           const std::string &filename, real_t scale)
    : fullResolution(resolution), diagonal(diagonal * .001),
      filter(std::move(filter)), filename(filename), scale(scale) {
  // compute film image bounds
  croppedPixelBounds = bounds2i(
      ponos::point2i(std::ceil(fullResolution.x * cropWindow.lower.x),
                     std::ceil(fullResolution.y * cropWindow.lower.y)),
      ponos::point2i(std::ceil(fullResolution.x * cropWindow.upper.x),
                     std::ceil(fullResolution.y * cropWindow.upper.y)));
  // allocate film image storage
  pixels = std::unique_ptr<Pixel[]>(new Pixel[croppedPixelBounds.area()]);
  // precompute filter weight table
  int offset = 0;
  for (int y = 0; y < filterTableWidth; ++y)
    for (int x = 0; x < filterTableWidth; ++x) {
      ponos::point2 p((x + 0.5f) * filter->radius.x / filterTableWidth,
                      (y + 0.5f) * filter->radius.y / filterTableWidth);
      filterTable[offset] = filter->evaluate(p);
    }
}

bounds2i Film::sampleBounds() const {
  bounds2 floatBounds(ponos::floor(ponos::point2(croppedPixelBounds.lower) +
                                   ponos::vec2(0.5) + filter->radius),
                      ponos::ceil(ponos::point2(croppedPixelBounds.upper) +
                                  ponos::vec2(0.5) - filter->radius));
  return (bounds2i)floatBounds;
}

bounds2 Film::physicalExtent() const {
  real_t aspect = (real_t)fullResolution.y / (real_t)fullResolution.x;
  real_t x = std::sqrt(diagonal * diagonal / (1 + aspect * aspect));
  real_t y = aspect * x;
  return bounds2(ponos::point2(-x / 2, -y / 2), ponos::point2(x / 2.y / 2));
}

std::unique_ptr<FilmTile> Film::filmTile(const bounds2i &sampleBounds) {
  // bound image pixels that samples in sampleBounds contribute to
  ponos::vec2 halfPixel = ponos::vec2(0.5f);
  bounds2 floatBounds = (bounds2)sampleBounds;
  ponos::point2i p0 = (ponos::point2i)ponos::ceil(floatBounds.lower -
                                                  halfPixel - filter->radius);
  ponos::point2i p1 = (ponos::point2i)ponos::floot(floatBounds.upper -
                                                   halfPixel + filter->radius) +
                      ponos::vec2i(1);
  bounds2i tilePixelBounds = intersect(bounds2i(p0, p1), croppedPixelBounds);
  return std::unique_ptr<FilmTile>(new filmTile(tilePixelBounds, filter->radius,
                                                filterTable, filterTableWidth));
}

void Film::mergeFilmTile(std::unique_ptr<FilmTile> tile) {
  std::lock_guard<std::mutex> lock(mutex);
  for (ponos::point2i pixel : tile->getPixelBounds()) {
    // merge pixel into Film::pixels
    const FilmTilePixel &tilePixel = tile->getPixel(pixel);
    Pixel &mergePixel = getPixel(pixel);
    real_t xyz[3];
    tilePixel.contribSum.toXYZ(xyz);
    for (int i = 0; i < 3; ++i)
      mergePixel.xyz[i] += xyz[i];
    mergePixel.filterWeightSum += tilePixel.filterWeightSum;
  }
}

Pixel &Film::getPixel(const ponos::point2i &p) {
  int width = croppedPixelBounds.upper.x - croppedPixelBounds.lower.x;
  int offset = (p.x - croppedPixelBounds.lower.x) +
               (p.y - croppedPixelBounds.lower.y) * width;
  return pixels[offset];
}

void Film::setImage(const Spectrum *img) const {
  int nPixels = croppedPixelBounds.area();
  for (int i = 0; i < nPixels; ++i) {
    Pixel &p = pixels[i];
    img[i].toXYZ(p.xyz);
    p.filterWeightSum = 1;
    p.splatXYZ[0] = p.splatXYZ[1] = p.splatXYZ[2] = 0;
  }
}

void Film::addSplat(const ponos::point2 &p, const Spectrum &v) {
  if (!insideExclusive((ponos::point2i)p, croppedPixelBounds))
    return;
  real_t xyz[3];
  v.toXYZ(xyz);
  Pixel &pixel = getPixel((ponos::point2i)p);
  for (int i = 0; i < 3; ++i)
    pixel.splatXYZ[i].add(xyz[i]);
}

} // namespace helios