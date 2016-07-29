#pragma once

#include <ponos.h>

namespace aergia {

	class Trackball {
		public:
			Trackball() {
				radius_ = 1.f;
			}

		protected:
			float radius_;
			float angle_;
	};

} // aergia namespace

