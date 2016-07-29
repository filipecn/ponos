#pragma once

#include "geometry/point.h"
#include "geometry/vector.h"

namespace ponos {

	class Line {
		public:
			Line() {}
			Line(ponos::Point3 a, ponos::vec3 d){
				a_ = a;
				d_ = normalize(d);
			}

			ponos::vec3 direction(){
				return normalize(d_);
			}

			ponos::Point3 point(float t){
				return a_ + d_ * t;
			}

			float projection(ponos::Point3 p){
				return dot((p - a_), d_);
			}

			ponos::Point3 closestPoint(ponos::Point3 p){
				return point(projection(p));
			}
			ponos::Point3 a_;
			ponos::vec3 d_;
	};

} // ponos namespace

