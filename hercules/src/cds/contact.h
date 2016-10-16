#ifndef HERCULES_CDS_CONTACT_H
#define HERCULES_CDS_CONTACT_H

#include <ponos.h>

namespace hercules {

	namespace cds {

		struct Contact2D {
			void* a;
			void* b;
			ponos::Point2 p;
			ponos::Normal2D n;
			ponos::vec2 ea, eb;
			bool vf;
			bool valid;
		};

	} // cds namespace

} // hercules namespace

#endif // HERCULES_CDS_CONTACT_H

