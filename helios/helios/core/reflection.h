#ifndef HELIOS_CORE_REFLECTION_H
#define HELIOS_CORE_REFLECTION_H

#include <helios/core/spectrum.h>
#include <ponos/geometry/vector.h>

namespace helios {

/// Reflections are computed in a coordinate system where the two tangent
/// vectors and the normal vector at the point being shaded are aligned with x,
/// y and z axes, respectively. All BRDF and BTDF work with vectors in this
/// system.
/// Spherical coordinates can be used to express directions here, where
/// **theta** is the angle between the direction and z axis, and **phi** is
/// formed with the x axis after projection onto the xy plane.
class ShadingCoordinateSystem {
public:
  /// \param w direction vector
  /// \return real_t cosine of angle theta (between w and z)
  static real_t cosTheta(const ponos::vec3 &w);
  /// \param w direction vector
  /// \return real_t squared cosine of angle theta (between w and z)
  static real_t cos2Theta(const ponos::vec3 &w);
  /// \param w direction vector
  /// \return real_t absolute cosine of angle theta (between w and z)
  static real_t absCosTheta(const ponos::vec3 &w);
  /// \param w direction vector
  /// \return real_t sine of angle theta (between w and z)
  static real_t sinTheta(const ponos::vec3 &w);
  /// \param w direction vector
  /// \return real_t squared sine of angle theta (between w and z)
  static real_t sin2Theta(const ponos::vec3 &w);
  /// \param w direction vector
  /// \return real_t tangent of angle theta (between w and z)
  static real_t tanTheta(const ponos::vec3 &w);
  /// \param w direction vector
  /// \return real_t squared tangent of angle theta (between w and z)
  static real_t tan2Theta(const ponos::vec3 &w);
  /// \param w direction vector
  /// \return real_t cosine of angle phi (between projected w and x)
  static real_t cosPhi(const ponos::vec3 &w);
  /// \param w direction vector
  /// \return real_t squared cosine of angle phi (between projected w and x)
  static real_t cos2Phi(const ponos::vec3 &w);
  /// \param w direction vector
  /// \return real_t sine of angle phi (between projected w and x)
  static real_t sinPhi(const ponos::vec3 &w);
  /// \param w direction vector
  /// \return real_t squared sine of angle phi (between projected w and x)
  static real_t sin2Phi(const ponos::vec3 &w);
  /// Computes the cosine of the angle between two vectors in the shading
  /// coordinate system
  /// \param wa vector (in the shading coordinate system)
  /// \param wb vector (in the shading coordinate system)
  /// \return real_t cosine value
  static real_t cosDPhi(const ponos::vec3 &wa, const ponos::vec3 &wb);
};

class BxDF {
public:
  /// Scattering types
  enum Type {
    BSDF_REFLECTION = 1 << 0,
    BSDF_TRANSMISSION = 1 << 1,
    BSDF_DIFFUSE = 1 << 2,
    BSDF_GLOSSY = 1 << 3,
    BSDF_SPECULAR = 1 << 4,
    BSDF_ALL = BSDF_DIFFUSE | BSDF_REFLECTION | BSDF_TRANSMISSION |
               BSDF_GLOSSY | BSDF_SPECULAR,
  };
  /// \param type scattering types
  BxDF(Type type);
  virtual ~BxDF() = default;
  /// \param t scattering types
  /// \return true if t is present
  bool matchesFlags(Type t) const;
  /// \param wo outgoing viewing direction
  /// \param wi incident light direction
  /// \return the value of the distribution function for the given pair of
  /// directions.
  virtual Spectrum f(const ponos::vec3 &wo, const ponos::vec3 &wi) const = 0;
  /// Computes the direction of the incident light given an outgoing direction
  /// \param wo **[in]** outgoing viewing direction
  /// \param wi **[out]**  computed incident light direction
  /// \param sample **[in]**
  /// \param pdf **[out]**
  /// \return the value of the distribution function for the given pair of
  /// directions.
  virtual Spectrum sample_f(const ponos::vec3 &wo, ponos::vec3 *wi,
                            const ponos::point2 &sample, real_t *pdf,
                            Type *sampledType = nullptr) const;
  /// Computes the hemispherical-directional reflectance that gives the total
  /// reflection in a given direction due to constant illumination over the
  /// hemisphere
  /// \param wo **[in]** outgoing direction
  /// \param nSamples number of samples
  /// \param samples hemisphere sample positions (only needed by some
  /// algorithms)
  /// \return Spectrum the value of the function
  virtual Spectrum rho(const ponos::vec3 &wo, int nSamples,
                       const ponos::point2 *samples) const;
  /// Computes the hemispherical-hemispherical reflectance that gives the
  /// fraction of incident light reflected by the surface when incident light is
  /// the same from all directions
  /// \param nSamples
  /// \param samples1
  /// \param samples2
  /// \return Spectrum
  virtual Spectrum rho(int nSamples, const ponos::point2 *samples1,
                       const ponos::point2 *samples2) const;

  const Type type; //!< scattering types
};

/// Wrapper for BxDF. Its spectrum values are scaled. Usefull to combine
/// multiple materials.
class ScaledBxDF : public BxDF {
public:
  /// \param bxdf bxdf object
  /// \param scale spectrum scale factor
  ScaledBxDF(BxDF *bxdf, const Spectrum &scale);
  ~ScaledBxDF() = default;
  /// \param wo outgoing viewing direction
  /// \param wi incident light direction
  /// \return the value of the distribution function for the given pair of
  /// directions scaled by **scale**.
  Spectrum f(const ponos::vec3 &wo, const ponos::vec3 &wi) const override;

private:
  BxDF *bxdf;     //!< BxDF object
  Spectrum scale; //!< spectrum scale factor
};

/// Provides an interface for computing Fresnel reflection coefficients.
/// _Fresnel equations_ describe the amount of light reflected from a surface
/// dependent on the direction.
class Fresnel {
public:
  /// Computes the Fresnel reflection formula for dielectric materials and
  /// unpolarized light.
  /// \param cosThetaI cosine of incident angle (formed with normal)
  /// \param etaI index of refraction for the incident media
  /// \param etaT index of refraction for the transmitted media
  /// \return real_t the Fresnel reflectance value
  static real_t frDielectric(real_t cosThetaI, real_t etaI, real_t etaT);
  /// Computes the Fresnel reflection formula for conductor materials. In this
  /// case, some incident light is potentially absorbed by the material and
  /// turned into heat presenting complex indices of refraction.
  /// \param cosThetaI cosine of incident angle (formed with normal)
  /// \param etaI index of refraction for the incident media
  /// \param etaT index of refraction for the transmitted media
  /// \param k absortion coefficient
  /// \return Spectrum
  static Spectrum frConductor(real_t cosThetaI, const Spectrum &etaI,
                              const Spectrum &etaT, const Spectrum &k);

public:
  /// Computes the amount of light reflected by the surface
  /// \param cosI cosine of incoming direction angle (with surface normal)
  /// \return Spectrum the amount of light reflected by the surface
  virtual Spectrum evaluate(real_t cosI) const = 0;
};

// Interface for conductor materials
class FresnelConductor : public Fresnel {
public:
  /// \param etaI index of refraction for the incident media
  /// \param etaT index of refraction for the transmitted media
  /// \param k absortion coefficient
  FresnelConductor(const Spectrum &etaI, const Spectrum &etaT,
                   const Spectrum &k);
  /// Computes the amount of light reflected by the surface
  /// \param cosI cosine of incoming direction angle (with surface normal)
  /// \return Spectrum the amount of light reflected by the surface
  Spectrum evaluate(real_t cosI) const override;

private:
  Spectrum etaI; //!< index of refraction for the incident media
  Spectrum etaT; //!< index of refraction for the transmitted media
  Spectrum k;    //!< absortion coefficient
};

// Interface for dielectric materials
class FresnelDielectric : public Fresnel {
public:
  /// \param etaI index of refraction for the incident media
  /// \param etaT index of refraction for the transmitted media
  FresnelDielectric(const real_t &etaI, const real_t &etaT);
  /// Computes the amount of light reflected by the surface
  /// \param cosI cosine of incoming direction angle (with surface normal)
  /// \return Spectrum the amount of light reflected by the surface
  Spectrum evaluate(real_t cosI) const override;

private:
  real_t etaI; //!< index of refraction for the incident media
  real_t etaT; //!< index of refraction for the transmitted media
};

} // namespace helios

#endif // HELIOS_CORE_REFLECTION_H