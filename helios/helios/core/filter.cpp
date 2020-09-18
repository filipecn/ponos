#include "helios/core/filter.h"

namespace helios {

Filter::Filter(const ponos::vec2 &radius)
    : radius(radius), invRadius(ponos::vec2(1 / radius.x, 1 / radius.y)) {}

BoxFilter::BoxFilter(const ponos::vec2 &radius) : Filter(radius) {}

real_t BoxFilter::evaluate(const ponos::point2 &p) const { return 1.; }

} // namespace helios
