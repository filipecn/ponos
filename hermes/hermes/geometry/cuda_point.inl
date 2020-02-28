template <typename T> __host__ __device__ Point2<T>::Point2() { x = y = 0.f; }

template <typename T> __host__ __device__ Point2<T>::Point2(T f) { x = y = f; }

template <typename T>
__host__ __device__ Point2<T>::Point2(T _x, T _y) : x(_x), y(_y) {}

template <typename T> __host__ __device__ T Point2<T>::operator[](int i) const {
  return (&x)[i];
}

template <typename T> __host__ __device__ T &Point2<T>::operator[](int i) {
  return (&x)[i];
}

template <typename T>
__host__ __device__ bool Point2<T>::operator==(const Point2 &p) const {
  return Check::is_equal(x, p.x) && Check::is_equal(y, p.y);
}

template <typename T>
Point2<T> Point2<T>::operator+(const Vector2<T> &v) const {
  return Point2(x + v.x, y + v.y);
}

template <typename T>
Point2<T> Point2<T>::operator-(const Vector2<T> &v) const {
  return Point2(x - v.x, y - v.y);
}

template <typename T>
__host__ __device__ Point2<T> Point2<T>::operator-(const T &f) const {
  return Point2(x - f, y - f);
}

template <typename T>
__host__ __device__ Point2<T> Point2<T>::operator+(const T &f) const {
  return Point2(x + f, y + f);
}

template <typename T>
__host__ __device__ Vector2<T> Point2<T>::operator-(const Point2 &p) const {
  return Vector2<T>(x - p.x, y - p.y);
};

template <typename T>
__host__ __device__ Point2<T> Point2<T>::operator/(T d) const {
  return Point2(x / d, y / d);
}

template <typename T>
__host__ __device__ Point2<T> Point2<T>::operator*(T f) const {
  return Point2(x * f, y * f);
}

template <typename T>
__host__ __device__ Point2<T> &Point2<T>::operator+=(const Vector2<T> &v) {
  x += v.x;
  y += v.y;
  return *this;
}

template <typename T>
__host__ __device__ Point2<T> &Point2<T>::operator-=(const Vector2<T> &v) {
  x -= v.x;
  y -= v.y;
  return *this;
}

template <typename T>
__host__ __device__ Point2<T> &Point2<T>::operator/=(T d) {
  x /= d;
  y /= d;
  return *this;
}

template <typename T>
__host__ __device__ bool Point2<T>::operator<(const Point2 &right) const {
  if (x < right.x)
    return true;
  else if (x > right.x)
    return false;
  if (y < right.y)
    return true;
  else if (y > right.y)
    return false;
  return false;
}

template <typename T>
__host__ __device__ bool Point2<T>::operator>=(const Point2 &p) const {
  return x >= p.x && y >= p.y;
}

template <typename T>
__host__ __device__ bool Point2<T>::operator<=(const Point2 &p) const {
  return x <= p.x && y <= p.y;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Point2<T> &p) {
  os << "[Point2] " << p.x << " " << p.y << std::endl;
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Point3<T> &p) {
  os << "[Point3] " << p.x << " " << p.y << " " << p.z << std::endl;
  return os;
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__ Point2<T> operator*(T f, const Point2<T> &p) {
  return p * f;
}

template <typename T> T distance(const Point2<T> &a, const Point2<T> &b) {
  return (a - b).length();
}

template <typename T> T distance2(const Point2<T> &a, const Point2<T> &b) {
  return (a - b).length2();
}
