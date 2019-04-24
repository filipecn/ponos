#ifndef POSEIDON_SIMULATION_CUDA_COLLIDER_H
#define POSEIDON_SIMULATION_CUDA_COLLIDER_H

namespace poseidon {

namespace cuda {

template <typename T> class Collider2 {
public:
  /// \param v
  void setVelocity(const hermes::cuda::Vector2<T> &v) { velocity = v; }
  /// \param p
  /// \return true
  /// \return false
  virtual __host__ __device__ bool
  intersect(const hermes::cuda::Point2<T> &p) const = 0;
  /// \param p
  /// \return T
  virtual __host__ __device__ T distance(const hermes::cuda::Point2<T> &p,
                                         hermes::cuda::Point2<T> *s) = 0;

  hermes::cuda::Vector2<T> velocity;
};

template <typename T> class Collider2Set : public Collider2<T> {
public:
  __host__ __device__ Collider2Set(Collider2<T> **l, size_t n)
      : list(l), n(n) {}
  __host__ __device__ bool
  intersect(const hermes::cuda::Point2<T> &p) const override {
    for (size_t i = 0; i < n; i++)
      if (list[i]->intersect(p))
        return true;
    return false;
  }
  __host__ __device__ T distance(const hermes::cuda::Point2<T> &p,
                                 hermes::cuda::Point2<T> *s) override {
    T mdist = hermes::cuda::Constants::greatest<T>();
    bool negative = false;
    hermes::cuda::Point2<T> cp;
    for (int i = 0; i < n; i++) {
      T dist = list[i]->distance(p, &cp);
      if (abs(dist) < mdist) {
        mdist = abs(dist);
        negative = dist < 0;
        *s = cp;
      }
    }
    return (negative ? -1 : 1) * mdist;
  }
  __host__ __device__ int size() const { return n; }

private:
  Collider2<T> **list;
  size_t n = 0;
};

template <typename T> class SphereCollider2 : public Collider2<T> {
public:
  __host__ __device__ SphereCollider2(const hermes::cuda::Point2<T> &center,
                                      T radius)
      : c(center), r(radius) {}
  __host__ __device__ bool
  intersect(const hermes::cuda::Point2<T> &p) const override {
    return hermes::cuda::distance2(c, p) <= r * r;
  }
  __host__ __device__ T distance(const hermes::cuda::Point2<T> &p,
                                 hermes::cuda::Point2<T> *s) override {
    return hermes::cuda::distance(c, p) - r;
  }

private:
  hermes::cuda::Point2<T> c;
  T r;
};

template <typename T> class BoxCollider2 : public Collider2<T> {
public:
  __host__ __device__ BoxCollider2(const hermes::cuda::BBox2<T> &box)
      : box(box) {}
  __host__ __device__ bool
  intersect(const hermes::cuda::Point2<T> &p) const override {
    return box.inside(p);
  }
  __host__ __device__ T distance(const hermes::cuda::Point2<T> &p,
                                 hermes::cuda::Point2<T> *s) override {
    return 0;
  }

private:
  hermes::cuda::BBox2<T> box;
};

} // namespace cuda

} // namespace poseidon

#endif // POSEIDON_SIMULATION_CUDA_COLLIDER_H