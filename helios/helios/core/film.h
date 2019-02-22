#ifndef HELIOS_CORE_FILM_H
#define HELIOS_CORE_FILM_H

#include <helios/core/filter.h>
#include <helios/core/spectrum.h>
#include <memory>

namespace helios {

/// Pixel structure stored in film tiles
struct FilmTilePixel {
  Spectrum contribSum =
      0.f; //!< sum of weighted contributions from pixel samples
  real_t filterWeightSum = 0.f; //!< sum of filter weights
};

/// Represents an image piece to be rendered
class FilmTile {
public:
  /// \param pixelBounds the bounds of the pixels in the final image
  /// \param filterRadius reconstruction filter radius
  /// \param filterTable reconstruction filter precomputed table
  /// \param filterTableSize precomputed table's size
  FilmTile(const bounds2i &pixelBounds, const ponos::vec2 &filterRadius,
           const real_t *filterTable, int filterTableSize);
  /// \param pFilm sample position
  /// \param L sample's radiance
  /// \param sampleWeight sample's filter weight
  void addSample(const ponos::point2 &pFilm, const Spectrum &L,
                 real_t sampleWeight = 1.);
  /// \param p pixel coordinates with respect to overall image
  /// \return FilmTilePixel& reference to pixel inside tile
  FilmTilePixel &getPixel(const ponos::point2i &p);
  /// \param p pixel coordinates with respect to overall image
  /// \return const FilmTilePixel& const reference to pixel inside tile
  const FilmTilePixel &getPixel(const ponos::point2i &p) const;
  /// \return bounds2i bound of pixels under the tile
  bounds2i getPixelBounds() const;

private:
  const bounds2i pixelBounds;     //!< bounds of the pixels in the final image
  const ponos::vec2 filterRadius; //!< reconstruction filter's radius
  const ponos::vec2 invFilterRadius;  //!< radius reciprocal
  const real_t *filterTable;          //!< pointer to filter's precomputed table
  const int filterTableSize;          //!< size of table
  std::vector<FilmTablePixel> pixels; //!< redered pixels
};

/// Dictates how the incident light is acctually transformed into colors in an
/// image.
class Film {
public:
  /// \param resolution image's size in pixels
  /// \param cropWindow specifies a subset of the image to render (NDC space)
  /// \param filter used to compute radiance contributions to each pixel
  /// \param diagonal length of the diagonal of the film's physical area
  /// (millimeters)
  /// \param filename path/to/output/image
  /// \param scale control how the pixel values are stored in files
  Film(const ponos::point2i &resolution, const bounds2 &cropWindow,
       std::unique_ptr<Filter> filter, real_t diagonal,
       const std::string &filename, real_t scale);
  /// Computes the range of pixel values for the Sampler
  /// \return bounds2i the area to be sampled
  bounds2i sampleBounds() const;
  /// \return bounds2 extent of the film in the scene
  bounds2 physicalExtent() const;
  /// \param sampleBounds tile's region
  /// \return std::unique_ptr<FilmTile> a pointer to a FilmTile object that
  /// stores contributions for the pixels in its region
  std::unique_ptr<FilmTile> filmTile(const bounds2i &sampleBounds);
  /// Note that ownership of tile is transferred to this method, so the caller
  /// should not attempt to add contributions to the tile after.
  /// \param tile tile unique reference
  void mergeFilmTile(std::unique_ptr<FilmTile> tile);
  /// Sets the entire image
  /// \param img pixel's XYZ values
  void setImage(const Spectrum *img) const;
  /// Splats contributions to arbitrary pixels
  /// \param p position
  /// \param v spectrum data
  void addSplat(const ponos::point2 &p, const Spectrum &v);

  const ponos::point2i fullResolution; //!< image's size in pixels
  bounds2i croppedPixelBounds; //!<  piece of image to be rendered/stored
  const real_t diagonal; //!< length of the diagonal of the film's physical area
                         //!< in meters
  std::unique_ptr<Filter>
      filter;                 //!< used to interpolate radiance values to pixels
  const std::string filename; //!< path/to/output/image

private:
  struct Pixel {
    real_t xyz[3] = {0, 0, 0};  //!< color in XYZ color space
    real_t filterWeightSum = 0; //!< sum of filter weights of radiance samples
    AtomicFloat splatXYZ[3];    //!< unweighted sum of samples splats
    real_t pad;                 //!< to ensure byte alignement
  };

  /// \param p pixel position
  /// \return Pixel& pixel's reference
  Pixel &Film::getPixel(const ponos::point2i &p);

  std::unique_ptr<Pixel[]> pixels; //!< image's pixel structures
  static constexpr int filterTableWidth =
      16; //!< Generally, every image sample contributes to 16 pixels in the
          //!< final image
  real_t filterTable[filterTableWidth *
                     filterTableWidth]; //!< precomputed table for filter values
                                        //!< to save computations of filte's
                                        //!< evaluate method. f = f(|x|, |y|)
  std::mutex mutex;                     //!< used when merging tiles
};

} // namespace helios

#endif // HELIOS_CORE_FILM_H
