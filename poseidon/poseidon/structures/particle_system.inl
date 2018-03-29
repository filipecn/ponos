template<typename T>
T ParticleSystem::minValue(size_t p) const {
  if (std::is_same<T, double>::value) {
    FATAL_ASSERT(p < propertiesD_.size());
    double M = INFINITY;
    pointSet_->iteratePoints([&](uint id, ponos::Point3 pos) {
      UNUSED_VARIABLE(pos);
      M = std::min(M, propertiesD_[p][id]);
    });
    return M;
  }
  return T(0);
}

template<typename T>
T ParticleSystem::maxValue(size_t p) const {
  if (std::is_same<T, double>::value) {
    FATAL_ASSERT(p < propertiesD_.size());
    double M = -INFINITY;
    pointSet_->iteratePoints([&](uint id, ponos::Point3 pos) {
      UNUSED_VARIABLE(pos);
      M = std::max(M, propertiesD_[p][id]);
    });
    return M;
  }
  return T(0);
}

template<typename T>
uint ParticleSystem::addProperty(T v) {
  if (std::is_same<T, double>::value) {
    propertiesD_.emplace_back(
        std::vector<double>((propertiesD_.size()) ? propertiesD_[0].size() : pointSet_->size(), v));
    return std::max(static_cast<size_t>(propertiesD_.size() - 1u), static_cast<size_t>(0));
  }
  return 0;
}

template<typename T>
T &poseidon::ParticleSystem::property(uint i, uint p) {
  if (std::is_same<T, double>::value) {
    FATAL_ASSERT(p < propertiesD_.size());
    if (i >= propertiesD_[p].size())
      propertiesD_[p].resize(i + 1);
    return propertiesD_[p][i];
  }
  return dummyD_;
}

template<typename T>
T ParticleSystem::gatherProperty(ponos::Point3 p,
                                 uint i, double r,
                                 ponos::InterpolatorInterface<ponos::Point3, T> *interpolator) {
  std::vector<ponos::Point3> points;
  std::vector<T> values;
  search(ponos::BBox(p, static_cast<float>(r)), [&](ParticleAccessor acc) {
    points.emplace_back(acc.position());
    values.emplace_back(acc.property<T>(i));
  });
  if (points.size() > 3)
    return interpolator->interpolateAt(p, points, values);
  return T(0.);
}

template<typename T>
std::vector<T> ParticleSystem::gatherProperties(ponos::Point3 p,
                                                const std::vector<size_t> &ids,
                                                double r,
                                                ponos::InterpolatorInterface<ponos::Point3, T> *interpolator) {
  std::vector<ponos::Point3> points;
  std::vector<std::vector<T>> values(ids.size());
  search(ponos::BBox(p, static_cast<float>(r)), [&](ParticleAccessor acc) {
    points.emplace_back(acc.position());
    for (size_t i = 0; i < ids.size(); i++)
      values[i].emplace_back(acc.property<T>(ids[i]));
  });
  std::vector<T> ans(ids.size(), T(0));
  if (points.size() > 3) {
    std::cerr << "solve " << points.size() << std::endl;
    for (size_t i = 0; i < ids.size(); i++)
      ans[i] = interpolator->interpolateAt(p, points, values[i]);
  }
  return ans;
}

