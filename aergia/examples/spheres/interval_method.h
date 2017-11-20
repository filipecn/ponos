#pragma once

#include <boost/numeric/interval.hpp>
using namespace boost::numeric;
using namespace boost::numeric::interval_lib;
#include "aaflib/aa.h"

template <class IntervalForm> class IntervalMethodInterface {
public:
  IntervalMethodInterface() {}
  virtual ~IntervalMethodInterface() {}

  IntervalForm form() const { return I; }

protected:
  IntervalForm I;
};

typedef interval<double, policies<save_state<rounded_transc_std<double>>,
                                  checking_base<double>>> IA_Interval;
class IAI : public IntervalMethodInterface<IA_Interval> {
public:
  IAI() {}
  IAI(IA_Interval a) { I = a; }
  IAI(double a, double b) { I = IA_Interval(a, b); }
  ~IAI() {}

  double lower() { return I.lower(); }
  double upper() { return I.upper(); }
  IAI operator+(const IAI &b) const { return IAI(I + b.form()); }
  IAI operator-(const IAI &b) const { return IAI(I - b.form()); }
  IAI operator*(const IAI &b) const { return IAI(I * b.form()); }
  IAI operator*(const double &b) const { return IAI(I * b); }
  IAI operator+(const double &b) const { return IAI(I + b); }
  IAI operator-(const double &b) const { return IAI(I - b); }
};

bool zero_in(const IAI &a) { return zero_in(a.form()); }
double width(const IAI &a) { return width(a.form()); }
IAI cos(const IAI &a) { return IAI(cos(a.form())); }
IAI sin(const IAI &a) { return IAI(sin(a.form())); }
IAI square(const IAI &a) { return IAI(square(a.form())); }
inline IAI operator*(double f, const IAI &v) { return IAI(f * v.form()); }

class AAI : public IntervalMethodInterface<AAF> {
public:
  AAI() {}
  AAI(double a, double b) { I = AAInterval(a, b); }
  AAI(AAF a) { I = a; }
  ~AAI() {}

  double lower() { return I.getMin(); }
  double upper() { return I.getMax(); }
  AAI operator+(const AAI &b) const { return AAI(I + b.form()); }
  AAI operator-(const AAI &b) const { return AAI(I - b.form()); }
  AAI operator*(const AAI &b) { return AAI(I * b.form()); }
  AAI operator*(const double &b) const { return AAI(b * I); }
  AAI operator+(const double &b) const { return AAI(I + b); }
  AAI operator-(const double &b) const { return AAI(I - b); }
};

bool zero_in(const AAI &a) {
  AAInterval I = a.form().convert();
  return I.getlo() <= 0.0 && I.gethi() >= 0.0;
}
double width(const AAI &a) { return a.form().convert().width(); }
AAI cos(const AAI &a) { return AAI(cos(a.form())); }
AAI sin(const AAI &a) { return AAI(sin(a.form())); }
AAI square(const AAI &a) { return AAI(square(a.form())); }
inline AAI operator*(double f, const AAI &v) { return AAI(f * v.form()); }
