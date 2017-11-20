#ifndef PONOS_SPATIAL_ARRAY_H
#define PONOS_SPATIAL_ARRAY_H

#include <ponos/geometry/numeric.h>
#include <ponos/geometry/ray.h>
#include <ponos/spatial/spatial_structure_interface.h>

#include <memory>
#include <vector>

namespace ponos {

template <typename ObjectType>
class Array : public SpatialStructureInterface<ObjectType> {
public:
  Array() {}
  virtual ~Array() {}
  /* @inherit */
  void add(ObjectType *o) override { objects.emplace_back(o); }
  /* @inherit */
  void iterate(std::function<void(const ObjectType *o)> f) const override {
    for (const auto e : objects) {
      f(e);
    }
  }
  /* @inherit */
  ObjectType *intersect(const ponos::Ray3 &r,
                        float *t = nullptr) const override {
    ObjectType *ret = nullptr;
    float mint = INFINITY, curT = INFINITY;
    for (auto e : objects) {
      if (e->intersect(r, &curT)) {
        if (mint > curT) {
          mint = curT;
          ret = e;
        }
      }
    }
    if (t != nullptr)
      *t = mint;
    return ret;
  }

private:
  std::vector<ObjectType *> objects;
};

} // ponos namespace

#endif // PONOS_SPATIAL_ARRAY_H
