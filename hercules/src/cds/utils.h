#ifndef HERCULES_CDS_UTILS_H
#define HERCULES_CDS_UTILS_H

#include "cds/contact.h"
#include "cds/gjk.h"

#include <ponos.h>

namespace hercules {

	namespace cds {

    inline void compute_circle_circle_collision(const ponos::Circle* c1, const ponos::Circle* c2, Contact2D& c,
                                                const ponos::Transform2D* t1 = nullptr, const ponos::Transform2D* t2 = nullptr) {
      ponos::Point2 center1 = (t1 != nullptr) ? (*t1)(c1->c) : c1->c;
      ponos::Point2 center2 = (t2 != nullptr) ? (*t2)(c2->c) : c2->c;
      float dist = ponos::distance(center1, center2);
      float pd = dist - c1->r - c2->r;
      c.valid = pd <= 0.f;
      if(c.valid) {
        ponos::vec2 d = (dist == 0.f)? ponos::vec2(1, 0) : ponos::normalize(center1 - center2);
        c.normal = ponos::Normal2D(d);
        c.points[0].position = ((center1 + d * c1->r) + ponos::vec2(center2 - d * c2->r)) * 0.5f;
        c.points[0].penetration = pd;
        c.pointCount = 1;
      }
    }

    inline Contact2D compute_collision(const ponos::Shape* s1, const ponos::Shape* s2,
    const ponos::Transform2D* t1 = nullptr, const ponos::Transform2D* t2 = nullptr) {
      Contact2D c;
      if(!s1 || !s2) {
        c.valid = false;
        return c;
      }
      if(s1->type == ponos::ShapeType::SPHERE && s2->type == ponos::ShapeType::SPHERE) {
        compute_circle_circle_collision(static_cast<const ponos::Circle*>(s1),
                                        static_cast<const ponos::Circle*>(s2), c, t1, t2);
      }
      else
      c.valid = GJK::intersect(*static_cast<const ponos::Polygon*>(s1),
                               *static_cast<const ponos::Polygon*>(s2), t1, t2);
      return c;
    }

  } // cds namespace

} // hercules namespace

#endif // HERCULES_CDS_AABB_SWEEP_H
