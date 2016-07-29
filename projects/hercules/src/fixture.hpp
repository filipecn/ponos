#pragma once

#include "shape.hpp"

#include <memory>

namespace hercules {

	class Fixture {
  	public:
	 		Fixture() {}

			std::shared_ptr<Shape> shape;
	};

} // hercules namespace

