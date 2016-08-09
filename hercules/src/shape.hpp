#pragma once

#include <ponos.h>
#include <vector>

namespace hercules {

	class Shape {
  	public:
	 		Shape() {}
			virtual ~Shape() {}
	};

	class Polygon : public Shape {
		public:
			std::vector<ponos::Point2> vertices;
	};

	class Circle : public Shape {
		public:
			float radius;
			ponos::Point2 center;
	};

} // hercules namespace

