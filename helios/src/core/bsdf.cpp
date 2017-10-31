#include "core/bsdf.h"

namespace helios {

    BSDF::BSDF(const DifferentialGeometry &dg, const ponos::Normal &ngeom, float e)
      : dgShading(dg), eta(e) {
        ng = ngeom;
        nn = dgShading.n;
        sn = ponos::normalize(dgShading.dpdu);
        tn = ponos::cross(ponos::vec3(nn.x, nn.y, nn.z), sn);
        nBxDFs = 0;
      }

    Spectrum BSDF::f(const ponos::vec3 &woW, const ponos::vec3 &wiW, BxDFType flags) const {
      ponos::vec3 wi = worldToLocal(wiW), wo = worldToLocal(woW);
      ponos::vec3 vng(ng);
      if(ponos::dot(wiW, vng) * ponos::dot(woW, vng) > 0)
        flags = BxDFType(flags & ~BSDF_TRANSMISSION);
      else
        flags = BxDFType(flags & ~BSDF_REFLECTION);
      Spectrum f = 0.;
      for(int i = 0; i < nBxDFs; i++)
        if(bxdfs[i]->matchesFlags(flags))
          f += bxdfs[i]->f(wo, wi);
      return f;
    }
} // helios namespace
