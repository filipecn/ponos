#include "2d/rigid_body.h"
#include "2d/fixture.h"
#include "2d/world.h"
#include "cds/cds.h"

#ifdef _WINDOWS
inline void fv() {
  ponos::Point2 *pp = new ponos::Point2[2];
  ponos::Point2 *pp2 = new ponos::Point2[2];
  ponos::Shape *s = new ponos::Shape[2];
  ponos::Shape *f = new ponos::Polygon[2];
  ponos::Polygon* p = new ponos::Polygon[2];
  ponos::Polygon* p2 = new ponos::Polygon[2];
}
static void(*fvf)() = fv;
#endif
