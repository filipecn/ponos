#ifndef HERCULES_CDS_CONTACT_H
#define HERCULES_CDS_CONTACT_H

#include <ponos.h>

namespace hercules {

	namespace cds {

    struct ContactPoint2D {
      ponos::Point2 position;
      float penetration;
    };

		struct Contact2D {
      Contact2D() {
        a = b = nullptr;
      }
			void* a;
			void* b;
      size_t pointCount;
      ContactPoint2D points[4]; // TODO change to 2
			ponos::Normal2D normal;

      ponos::Point2 p;
			ponos::vec2 ea, eb;
      bool vf;
			bool valid;
		};

	} // cds namespace

} // hercules namespace

#endif // HERCULES_CDS_CONTACT_H
