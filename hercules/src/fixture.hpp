#ifndef HERCULES_FIXTURE_H
#define HERCULES_FIXTURE_H

#include "shape.hpp"

#include <memory>

namespace hercules {

	class Fixture {
  	public:
	 		Fixture() {}

			std::shared_ptr<Shape> shape;
	};

} // hercules namespace

#endif
