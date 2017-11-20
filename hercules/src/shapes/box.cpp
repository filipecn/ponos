#include "shapes/box.h"

namespace hercules {

	Box::Box(float w, float h, float d) {
		density = 1.f;
		mass = w * h * d * density;
		ponos::mat3 Ibody = (mass / 12.f) *
			ponos::mat3((h * h + d * d),             0.f,             0.f,
															0.f, (w * w + d * d),             0.f,
															0.f,             0.f, (w * w + h * h));
		IbodyInv = ponos::inverse(Ibody);
	}

} // hercules namespace
