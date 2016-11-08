#include "core/material.h"

namespace helios {

  void Material::bump(const std::shared_ptr<Texture<float> > &d,
    const DifferentialGeometry &dgGeom,
    const DifferentialGeometry &dgs,
    DifferentialGeometry *dgBump) {
      // compute offset
      DifferentialGeometry dgEval = dgs;
      float du = .5f * (fabsf(dgs.dudx) + fabsf(dgs.dudy));
      if(du == 0.f) du = .01f;
      dgEval.p = dgs.p + du * dgs.dpdu;
      dgEval.u = dgs.u + du;
      dgEval.nn = ponos::normalize((ponos::Normal)ponos::cross(dgs.dpdu, dgs.dpdv) + du * dgs.dndu);
      float uDisplace = d->evaluate(dgEval);

      float dv = .5f * (fabsf(dgs.dvdx) + fabsf(dgs.dvdy));
      if(dv == 0.f) dv = .01f;
      dgEval.p = dgs.p + dv * dgs.dpdv;
      dgEval.u = dgs.u;
      dgEval.v = dgs.v + dv;
      dgEval.nn = ponos::normalize((ponos::Normal)ponos::cross(dgs.dpdu, dgs.dpdv) + dv * dgs.dndv);
      float vDisplace = d->evaluate(dgEval);
      float displace = d->evaluate(dgs);
      // compute bump-mapped...
      *dgBump = dgs;
      dgBump->dpdu = dgs.dpdu + (uDisplace - displace) / du * ponos::vec3(dgs.nn) + displace * ponos::vec3(dgs.dndu);
      dgBump->dpdv = dgs.dpdv + (vDisplace - displace) / dv * ponos::vec3(dgs.nn) + displace * ponos::vec3(dgs.dndv);
      dgBump->nn = ponos::Normal(ponos::normalize(ponos::cross(dgBump->dpdu, dgBump->dpdv)));
      if(dgs.shape->reverseOrientation ^ dgs.shape->transformSwapsHandedness)
        dgBump->nn *- -1.f;
      // orient shading
      dgBump->nn = faceForward(dgBump->nn, dgGeom.nn);
    }
} // helios namespace
