#pragma once

#include "geometry/point.h"
#include "geometry/vector.h"

namespace ponos {

  class Ray {
  public:
    Ray();
    Ray(const Point3& origin, const Vector3& direction,
        float start = 0.f, float end = INFINITY, float t = 0.f, int d = 0);
    Ray(const Point3& origin, const Vector3& direction, const Ray& parent,
        float start, float end = INFINITY);
    Point3 operator()(float t) const {
      return o + d * t;
    }
    Point3 o;
    Vector3 d;
    mutable float min_t, max_t;
    float time;
    int depth;
  };

  typedef Ray ray;

}; // ponos namespace
