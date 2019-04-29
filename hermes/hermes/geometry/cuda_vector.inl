
template <typename T> __host__ __device__ Vector2<T>::Vector2() { x = y = 0; }

template <typename T>
__host__ __device__ Vector2<T>::Vector2(T _x, T _y) : x(_x), y(_y) {}

template <typename T>
__host__ __device__ Vector2<T>::Vector2(const Point2<T> &p) : x(p.x), y(p.y) {}

template <typename T> __host__ __device__ Vector2<T>::Vector2(T f) {
  x = y = f;
}

template <typename T> __host__ __device__ Vector2<T>::Vector2(T *f) {
  x = f[0];
  y = f[1];
}

template <typename T>
__host__ __device__ T Vector2<T>::operator[](size_t i) const {
  return (&x)[i];
}

template <typename T> __host__ __device__ T &Vector2<T>::operator[](size_t i) {
  return (&x)[i];
}

// arithmetic
template <typename T>
__host__ __device__ Vector2<T> Vector2<T>::
operator+(const Vector2<T> &v) const {
  return Vector2(x + v.x, y + v.y);
}

template <typename T>
__host__ __device__ Vector2<T> &Vector2<T>::operator+=(const Vector2<T> &v) {
  x += v.x;
  y += v.y;
  return *this;
}

template <typename T>
__host__ __device__ Vector2<T> Vector2<T>::
operator-(const Vector2<T> &v) const {
  return Vector2(x - v.x, y - v.y);
}

template <typename T>
__host__ __device__ Vector2<T> &Vector2<T>::operator-=(const Vector2<T> &v) {
  x -= v.x;
  y -= v.y;
  return *this;
}

template <typename T>
__host__ __device__ Vector2<T> Vector2<T>::operator*(T f) const {
  return Vector2(x * f, y * f);
}

template <typename T>
__host__ __device__ Vector2<T> &Vector2<T>::operator*=(T f) {
  x *= f;
  y *= f;
  return *this;
}

template <typename T>
__host__ __device__ Vector2<T> Vector2<T>::operator/(T f) const {
  Check::isEqual(f, 0.f);
  T inv = 1.f / f;
  return Vector2(x * inv, y * inv);
}

template <typename T>
__host__ __device__ Vector2<T> &Vector2<T>::operator/=(T f) {
  Check::isEqual(f, 0.f);
  T inv = 1.f / f;
  x *= inv;
  y *= inv;
  return *this;
}

template <typename T>
__host__ __device__ Vector2<T> Vector2<T>::operator-() const {
  return Vector2(-x, -y);
}

template <typename T>
__host__ __device__ bool Vector2<T>::operator==(const Vector2<T> &v) {
  return Check::isEqual(x, v.x) && Check::isEqual(y, v.y);
}

template <typename T>
__host__ __device__ bool Vector2<T>::operator!=(const Vector2<T> &v) {
  return !(Check::isEqual(x, v.x) && Check::isEqual(y, v.y));
}

template <typename T> __host__ __device__ T Vector2<T>::length2() const {
  return x * x + y * y;
}

template <typename T> __host__ __device__ T Vector2<T>::length() const {
  return sqrtf(length2());
}

template <typename T> __host__ __device__ Vector2<T> Vector2<T>::right() const {
  return Vector2(y, -x);
}

template <typename T> __host__ __device__ Vector2<T> Vector2<T>::left() const {
  return Vector2(-y, x);
}

template <typename T>
__host__ std::ostream &operator<<(std::ostream &os, const Vector2<T> &v) {
  os << "[vector2]" << v.x << " " << v.y << std::endl;
  return os;
}
