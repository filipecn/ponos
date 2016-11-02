#ifndef HERCULES_CDS_UTILS_H
#define HERCULES_CDS_UTILS_H

#include "cds/contact.h"
#include "cds/gjk.h"

#include <ponos.h>

namespace hercules {

	namespace cds {

    inline Contact2D compute_collision(const ponos::Shape* s1, const ponos::Shape* s2) {
      Contact2D c;
      if(!s1 || !s2) {
        c.valid = false;
        return c;
      }
      
      c.valid = GJK::intersect(*static_cast<const ponos::Polygon*>(s1),
                               *static_cast<const ponos::Polygon*>(s2));
      return c;
    }

  } // cds namespace

} // hercules namespace

#endif // HERCULES_CDS_AABB_SWEEP_H
