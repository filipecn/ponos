#pragma once

#include "fixture.hpp"

#include <memory>

namespace hercules {

	class Body {
		public:
			Body() {}

			std::shared_ptr<Fixture> fixture;
	};

} // hercules namespace

