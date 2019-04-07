
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
  CHECK_FLOAT_EQUAL(f, 0.f);
  T inv = 1.f / f;
  return Vector2(x * inv, y * inv);
}

template <typename T>
__host__ __device__ Vector2<T> &Vector2<T>::operator/=(T f) {
  CHECK_FLOAT_EQUAL(f, 0.f);
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
__host__ __device__ std::ostream &operator<<(std::ostream &os,
                                             const Vector2<T> &v) {
  os << "[vector2]" << v.x << " " << v.y << std::endl;
  return os;
}

template <typename T> __host__ __device__ Vector3<T>::Vector3() {
  x = y = z = 0;
}

template <typename T>
__host__ __device__ Vector3<T>::Vector3(T _f) : x(_f), y(_f), z(_f) {}

template <typename T>
__host__ __device__ Vector3<T>::Vector3(T _x, T _y, T _z)
    : x(_x), y(_y), z(_z) {}

template <typename T> __host__ __device__ Vector3<T>::Vector3(const T *v) {
  x = v[0];
  y = v[1];
  z = v[2];
}

template <typename T>
__host__ __device__ Vector3<T>::Vector3(const Point3<T> &p)
    : x(p.x), y(p.y), z(p.z) {}

template <typename T>
__host__ __device__ bool Vector3<T>::operator==(const Vector3<T> &v) const {
  return Check::isEqual(x, v.x) && Check::isEqual(y, v.y) &&
         Check::isEqual(z, v.z);
}

template <typename T>
__host__ __device__ bool Vector3<T>::operator<(const Vector3<T> &v) const {
  if (x < v.x)
    return true;
  if (y < v.y)
    return true;
  return z < v.z;
}

template <typename T>
__host__ __device__ bool Vector3<T>::operator>(const Vector3<T> &v) const {
  if (x > v.x)
    return true;
  if (y > v.y)
    return true;
  return z > v.z;
}

template <typename T>
__host__ __device__ T Vector3<T>::operator[](int i) const {
  return (&x)[i];
}

template <typename T> __host__ __device__ T &Vector3<T>::operator[](int i) {
  return (&x)[i];
}

template <typename T>
__host__ __device__ Vector2<T> Vector3<T>::xy(int i, int j) const {
  T a = x;
  if (i == 1)
    a = y;
  else if (i == 2)
    a = z;
  T b = y;
  if (j == 0)
    b = x;
  else if (j == 2)
    b = z;
  return Vector2<T>(a, b);
}

template <typename T>
__host__ __device__ Vector3<T> &Vector3<T>::operator=(const T &v) {
  x = y = z = v;
  return *this;
}

template <typename T>
__host__ __device__ Vector3<T> Vector3<T>::
operator+(const Vector3<T> &v) const {
  return Vector3(x + v.x, y + v.y, z + v.z);
}

template <typename T>
__host__ __device__ Vector3<T> &Vector3<T>::operator+=(const Vector3<T> &v) {
  x += v.x;
  y += v.y;
  z += v.z;
  return *this;
}

template <typename T>
__host__ __device__ Vector3<T> Vector3<T>::
operator-(const Vector3<T> &v) const {
  return Vector3(x - v.x, y - v.y, z - v.z);
}

template <typename T>
__host__ __device__ Vector3<T> &Vector3<T>::operator-=(const Vector3<T> &v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;
  return *this;
}

template <typename T>
__host__ __device__ Vector3<T> Vector3<T>::
operator*(const Vector3<T> &v) const {
  return Vector3(x * v.x, y * v.y, z * v.z);
}

template <typename T>
__host__ __device__ Vector3<T> Vector3<T>::operator*(T f) const {
  return Vector3(x * f, y * f, z * f);
}

template <typename T>
__host__ __device__ Vector3<T> &Vector3<T>::operator*=(T f) {
  x *= f;
  y *= f;
  z *= f;
  return *this;
}

template <typename T>
__host__ __device__ Vector3<T> Vector3<T>::operator/(T f) const {
  CHECK_FLOAT_EQUAL(f, 0.f);
  T inv = 1.f / f;
  return Vector3(x * inv, y * inv, z * inv);
}

template <typename T>
__host__ __device__ Vector3<T> &Vector3<T>::operator/=(T f) {
  CHECK_FLOAT_EQUAL(f, 0.f);
  T inv = 1.f / f;
  x *= inv;
  y *= inv;
  z *= inv;
  return *this;
}

template <typename T>
__host__ __device__ Vector3<T> &Vector3<T>::operator/=(const Vector3<T> &v) {
  x /= v.x;
  y /= v.y;
  z /= v.z;
  return *this;
}

template <typename T>
__host__ __device__ Vector3<T> Vector3<T>::operator-() const {
  return Vector3(-x, -y, -z);
}

template <typename T>
__host__ __device__ bool Vector3<T>::operator>=(const Vector3<T> &p) const {
  return x >= p.x && y >= p.y && z >= p.z;
}

template <typename T>
__host__ __device__ bool Vector3<T>::operator<=(const Vector3<T> &p) const {
  return x <= p.x && y <= p.y && z <= p.z;
}

template <typename T> __host__ __device__ T Vector3<T>::length2() const {
  return x * x + y * y + z * z;
}

template <typename T> __host__ __device__ T Vector3<T>::length() const {
  return sqrtf(length2());
}

template <typename T>
__host__ std::ostream &operator<<(std::ostream &os, const Vector3<T> &v) {
  os << "[vector3]" << v.x << " " << v.y << " " << v.z << std::endl;
  return os;
}
