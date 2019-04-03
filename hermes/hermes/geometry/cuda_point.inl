template <typename T> __host__ __device__ Point2<T>::Point2() { x = y = 0.f; }

template <typename T> __host__ __device__ Point2<T>::Point2(T f) { x = y = f; }

template <typename T>
__host__ __device__ Point2<T>::Point2(const T *v) : x(v[0]), y(v[1]) {}

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
  return Check::isEqual(x, p.x) && Check::isEqual(y, p.y);
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
__host__ __device__ bool Point2<T>::operator<(const Point2 &p) const {
  if (x >= p.x || y >= p.y)
    return false;
  return true;
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

template <typename T> __host__ __device__ Point3<T>::Point3() {
  x = y = z = 0.0f;
}

template <typename T>
Point3<T>::Point3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

template <typename T>
Point3<T>::Point3(const Vector3<T> &v) : x(v.x), y(v.y), z(v.z) {}

template <typename T>
Point3<T>::Point3(const Point2<T> &p) : x(p.x), y(p.y), z(0) {}

template <typename T>
Point3<T>::Point3(const T *v) : x(v[0]), y(v[1]), z(v[2]) {}

template <typename T> Point3<T>::Point3(T v) : x(v), y(v), z(v) {}

template <typename T> __host__ __device__ T Point3<T>::operator[](int i) const {
  return (&x)[i];
}

template <typename T> __host__ __device__ T &Point3<T>::operator[](int i) {
  return (&x)[i];
}

// arithmetic
template <typename T>
Point3<T> Point3<T>::operator+(const Vector3<T> &v) const {
  return Point3(x + v.x, y + v.y, z + v.z);
}

template <typename T>
__host__ __device__ Point3<T> Point3<T>::operator+(const Point3<T> &v) const {
  return Point3(x + v.x, y + v.y, z + v.z);
}

template <typename T>
__host__ __device__ Point3<T> Point3<T>::operator+(const T &f) const {
  return Point3(x + f, y + f, z + f);
}

template <typename T>
__host__ __device__ Point3<T> Point3<T>::operator-(const T &f) const {
  return Point3(x - f, y - f, z - f);
}

template <typename T>
__host__ __device__ Point3<T> &Point3<T>::operator+=(const Vector3<T> &v) {
  x += v.x;
  y += v.y;
  z += v.z;
  return *this;
}

template <typename T>
__host__ __device__ Vector3<T> Point3<T>::operator-(const Point3 &p) const {
  return Vector3<T>(x - p.x, y - p.y, z - p.z);
}

template <typename T>
Point3<T> Point3<T>::operator-(const Vector3<T> &v) const {
  return Point3(x - v.x, y - v.y, z - v.z);
}

template <typename T>
__host__ __device__ Point3<T> &Point3<T>::operator-=(const Vector3<T> &v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;
  return *this;
}

template <typename T>
__host__ __device__ bool Point3<T>::operator==(const Point3 &p) const {
  return Check::isEqual(p.x, x) && Check::isEqual(p.y, y) &&
         Check::isEqual(p.z, z);
}

template <typename T>
__host__ __device__ bool Point3<T>::operator>=(const Point3 &p) const {
  return x >= p.x && y >= p.y && z >= p.z;
}

template <typename T>
__host__ __device__ bool Point3<T>::operator<=(const Point3 &p) const {
  return x <= p.x && y <= p.y && z <= p.z;
}

template <typename T>
__host__ __device__ Point3<T> Point3<T>::operator*(T d) const {
  return Point3(x * d, y * d, z * d);
}

template <typename T>
__host__ __device__ Point3<T> Point3<T>::operator/(T d) const {
  return Point3(x / d, y / d, z / d);
}

template <typename T>
__host__ __device__ Point3<T> &Point3<T>::operator/=(T d) {
  x /= d;
  y /= d;
  z /= d;
  return *this;
}

template <typename T>
__host__ __device__ bool Point3<T>::operator==(const Point3 &p) {
  return Check::isEqual(x, p.x) && Check::isEqual(y, p.y) &&
         Check::isEqual(z, p.z);
}

template <typename T> __host__ __device__ Point2<T> Point3<T>::xy() const {
  return Point2<T>(x, y);
}

template <typename T> __host__ __device__ Point2<T> Point3<T>::yz() const {
  return Point2<T>(y, z);
}

template <typename T> __host__ __device__ Point2<T> Point3<T>::xz() const {
  return Point2<T>(x, z);
}

template <typename T>
__host__ __device__ Vector3<T> Point3<T>::asVector3() const {
  return Vector3<T>(x, y, z);
}

template <typename T> __host__ __device__ vec3i Point3<T>::asIVec3() const {
  return vec3i(static_cast<const int &>(x), static_cast<const int &>(y),
               static_cast<const int &>(z));
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Point3<T> &p) {
  os << "[Point3] " << p.x << " " << p.y << " " << p.z << std::endl;
  return os;
}

template <typename T>
__host__ __device__ Point3<T> &Point3<T>::operator*=(T d) {
  x *= d;
  y *= d;
  z *= d;
  return *this;
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

template <typename T> T distance(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length();
}

template <typename T> T distance2(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length2();
}