#ifndef HELIOS_CORE_BSDF_H
#define HELIOS_CORE_BSDF_H

#include "core/bxdf.h"
#include "core/differential_geometry.h"
#include "core/spectrum.h"

#include <ponos.h>

namespace helios {

#define BSDF_ALLOC(arena, Type) new (arena.Alloc(sizeof(Type))) Type

  /*  collection
   *  represents a collection of BRDFs and BTDFs.
   */
  class BSDF {
  public:
    BSDF(const DifferentialGeometry &dg, const ponos::Normal &ngeom, float e);

    #define MAX_BxDFS 8
    void add(BxDF *b) {
      ASSERT(nBxDFs < MAX_BxDFS);
      bxdfs[nBxDFs++] = b;
    }
    int numComponents() const { return nBxDFs; }
    int numComponents(BxDFType flags) const;
    ponos::vec3 worldToLocal(const ponos::vec3 &v) const {
      return ponos::vec3(ponos::dot(v, sn), ponos::dot(v, tn), ponos::dot(v, ponos::vec3(nn)));
    }
    ponos::vec3 localToWorld(const ponos::vec3 &v) const {
      return ponos::vec3(sn.x * v.x + tn.x * v.y + nn.x * v.z,
                         sn.y * v.x + tn.y * v.y + nn.y * v.z,
                         sn.z * v.x + tn.z * v.y + nn.z * v.z);
    }
    Spectrum f(const ponos::vec3 &woW, const ponos::vec3 &wiW, BxDFType flags) const;
    Spectrum rho(ponos::RNG &rng, BxDFType flags = BSDF_ALL, int sqrtSamples = 6) const;
    Spectrum rho(const ponos::vec3 &wo, ponos::RNG &rng, BxDFType flags = BSDF_ALL, int sqrtSamples = 6) const;

    const DifferentialGeometry dgShading;
    const float eta;

  private:
    ~BSDF() {}

    ponos::Normal nn, ng;
    ponos::vec3 sn, tn;
    int nBxDFs;
    BxDF *bxdfs[MAX_BxDFS];
  };

} // helios namespace

#endif
