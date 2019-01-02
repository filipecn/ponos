
template <typename T> Vector2<T>::Vector2() { x = y = 0; }

template <typename T> Vector2<T>::Vector2(T _x, T _y) : x(_x), y(_y) {
  ASSERT(!HasNaNs());
}

template <typename T>
Vector2<T>::Vector2(const Point2<T> &p) : x(p.x), y(p.y) {}

template <typename T>
Vector2<T>::Vector2(const Normal2<T> &n) : x(n.x), y(n.y) {}

template <typename T> Vector2<T>::Vector2(T f) { x = y = f; }

template <typename T> Vector2<T>::Vector2(T *f) {
  x = f[0];
  y = f[1];
}

template <typename T> T Vector2<T>::operator[](size_t i) const {
  ASSERT(i >= 0 && i <= 1);
  return (&x)[i];
}

template <typename T> T &Vector2<T>::operator[](size_t i) {
  ASSERT(i >= 0 && i <= 1);
  return (&x)[i];
}

// arithmetic
template <typename T>
Vector2<T> Vector2<T>::operator+(const Vector2<T> &v) const {
  return Vector2(x + v.x, y + v.y);
}

template <typename T> Vector2<T> &Vector2<T>::operator+=(const Vector2<T> &v) {
  x += v.x;
  y += v.y;
  return *this;
}

template <typename T>
Vector2<T> Vector2<T>::operator-(const Vector2<T> &v) const {
  return Vector2(x - v.x, y - v.y);
}

template <typename T> Vector2<T> &Vector2<T>::operator-=(const Vector2<T> &v) {
  x -= v.x;
  y -= v.y;
  return *this;
}

template <typename T> Vector2<T> Vector2<T>::operator*(T f) const {
  return Vector2(x * f, y * f);
}

template <typename T> Vector2<T> &Vector2<T>::operator*=(T f) {
  x *= f;
  y *= f;
  return *this;
}

template <typename T> Vector2<T> Vector2<T>::operator/(T f) const {
  CHECK_FLOAT_EQUAL(f, 0.f);
  T inv = 1.f / f;
  return Vector2(x * inv, y * inv);
}

template <typename T> Vector2<T> &Vector2<T>::operator/=(T f) {
  CHECK_FLOAT_EQUAL(f, 0.f);
  T inv = 1.f / f;
  x *= inv;
  y *= inv;
  return *this;
}

template <typename T> Vector2<T> Vector2<T>::operator-() const {
  return Vector2(-x, -y);
}

template <typename T> bool Vector2<T>::operator==(const Vector2<T> &v) {
  return IS_EQUAL(x, v.x) && IS_EQUAL(y, v.y);
}

template <typename T> T Vector2<T>::length2() const { return x * x + y * y; }

template <typename T> T Vector2<T>::length() const { return sqrtf(length2()); }

template <typename T> Vector2<T> Vector2<T>::right() const {
  return Vector2(y, -x);
}

template <typename T> Vector2<T> Vector2<T>::left() const {
  return Vector2(-y, x);
}

template <typename T> bool Vector2<T>::HasNaNs() const {
  return std::isnan(x) || std::isnan(y);
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Vector2<T> &v) {
  os << "[vector2]" << v.x << " " << v.y << std::endl;
  return os;
}

template <typename T> Vector3<T>::Vector3() { x = y = z = 0; }

template <typename T> Vector3<T>::Vector3(T _f) : x(_f), y(_f), z(_f) {
  ASSERT(!HasNaNs());
}

template <typename T>
Vector3<T>::Vector3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {
  ASSERT(!HasNaNs());
}

template <typename T> Vector3<T>::Vector3(const T *v) {
  x = v[0];
  y = v[1];
  z = v[2];
}

template <typename T>
Vector3<T>::Vector3(const Normal3<T> &n) : x(n.x), y(n.y), z(n.z) {}

template <typename T>
Vector3<T>::Vector3(const Point3<T> &p) : x(p.x), y(p.y), z(p.z) {}

template <typename T> bool Vector3<T>::operator==(const Vector3<T> &v) const {
  return IS_EQUAL(x, v.x) && IS_EQUAL(y, v.y) && IS_EQUAL(z, v.z);
}

template <typename T> bool Vector3<T>::operator<(const Vector3<T> &v) const {
  if (x < v.x)
    return true;
  if (y < v.y)
    return true;
  return z < v.z;
}

template <typename T> bool Vector3<T>::operator>(const Vector3<T> &v) const {
  if (x > v.x)
    return true;
  if (y > v.y)
    return true;
  return z > v.z;
}

template <typename T> T Vector3<T>::operator[](int i) const {
  ASSERT(i >= 0 && i <= 2);
  return (&x)[i];
}

template <typename T> T &Vector3<T>::operator[](int i) {
  ASSERT(i >= 0 && i <= 2);
  return (&x)[i];
}

template <typename T> Vector2<T> Vector3<T>::xy() const {
  return Vector2<T>(x, y);
}

template <typename T> Vector3<T> &Vector3<T>::operator=(const T &v) {
  x = y = z = v;
  return *this;
}

template <typename T>
Vector3<T> Vector3<T>::operator+(const Vector3<T> &v) const {
  return Vector3(x + v.x, y + v.y, z + v.z);
}

template <typename T> Vector3<T> &Vector3<T>::operator+=(const Vector3<T> &v) {
  x += v.x;
  y += v.y;
  z += v.z;
  return *this;
}

template <typename T>
Vector3<T> Vector3<T>::operator-(const Vector3<T> &v) const {
  return Vector3(x - v.x, y - v.y, z - v.z);
}

template <typename T> Vector3<T> &Vector3<T>::operator-=(const Vector3<T> &v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;
  return *this;
}

template <typename T>
Vector3<T> Vector3<T>::operator*(const Vector3<T> &v) const {
  return Vector3(x * v.x, y * v.y, z * v.z);
}

template <typename T> Vector3<T> Vector3<T>::operator*(T f) const {
  return Vector3(x * f, y * f, z * f);
}

template <typename T> Vector3<T> &Vector3<T>::operator*=(T f) {
  x *= f;
  y *= f;
  z *= f;
  return *this;
}

template <typename T> Vector3<T> Vector3<T>::operator/(T f) const {
  CHECK_FLOAT_EQUAL(f, 0.f);
  T inv = 1.f / f;
  return Vector3(x * inv, y * inv, z * inv);
}

template <typename T> Vector3<T> &Vector3<T>::operator/=(T f) {
  CHECK_FLOAT_EQUAL(f, 0.f);
  T inv = 1.f / f;
  x *= inv;
  y *= inv;
  z *= inv;
  return *this;
}

template <typename T> Vector3<T> &Vector3<T>::operator/=(const Vector3<T> &v) {
  x /= v.x;
  y /= v.y;
  z /= v.z;
  return *this;
}

template <typename T> Vector3<T> Vector3<T>::operator-() const {
  return Vector3(-x, -y, -z);
}

template <typename T> bool Vector3<T>::operator>=(const Vector3<T> &p) const {
  return x >= p.x && y >= p.y && z >= p.z;
}

template <typename T> bool Vector3<T>::operator<=(const Vector3<T> &p) const {
  return x <= p.x && y <= p.y && z <= p.z;
}

template <typename T> T Vector3<T>::length2() const {
  return x * x + y * y + z * z;
}

template <typename T> T Vector3<T>::length() const { return sqrtf(length2()); }

template <typename T> bool Vector3<T>::HasNaNs() const {
  return std::isnan(x) || std::isnan(y) || std::isnan(z);
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Vector3<T> &v) {
  os << "[vector3]" << v.x << " " << v.y << " " << v.z << std::endl;
  return os;
}

template <typename T> Vector4<T>::Vector4() { x = y = z = w = 0.0f; }

template <typename T>
Vector4<T>::Vector4(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {
  ASSERT(!HasNaNs());
}

template <typename T> T Vector4<T>::operator[](int i) const {
  ASSERT(i >= 0 && i <= 3);
  return (&x)[i];
}

template <typename T> T &Vector4<T>::operator[](int i) {
  ASSERT(i >= 0 && i <= 3);
  return (&x)[i];
}

template <typename T> Vector2<T> Vector4<T>::xy() { return Vector2<T>(x, y); }

template <typename T> Vector3<T> Vector4<T>::xyz() {
  return Vector3<T>(x, y, z);
}

template <typename T>
Vector4<T> Vector4<T>::operator+(const Vector4<T> &v) const {
  return Vector4(x + v.x, y + v.y, z + v.z, w + v.w);
}

template <typename T> Vector4<T> &Vector4<T>::operator+=(const Vector4<T> &v) {
  x += v.x;
  y += v.y;
  z += v.z;
  w += v.w;
  return *this;
}

template <typename T>
Vector4<T> Vector4<T>::operator-(const Vector4<T> &v) const {
  return Vector4(x - v.x, y - v.y, z - v.z, w - v.w);
}

template <typename T> Vector4<T> &Vector4<T>::operator-=(const Vector4<T> &v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;
  w -= v.w;
  return *this;
}

template <typename T> Vector4<T> Vector4<T>::operator*(T f) const {
  return Vector4(x * f, y * f, z * f, w * f);
}

template <typename T> Vector4<T> &Vector4<T>::operator*=(T f) {
  x *= f;
  y *= f;
  z *= f;
  w *= f;
  return *this;
}

template <typename T> Vector4<T> Vector4<T>::operator/(T f) const {
  CHECK_FLOAT_EQUAL(f, 0.f);
  T inv = 1.f / f;
  return Vector4(x * inv, y * inv, z * inv, w * inv);
}

template <typename T> Vector4<T> &Vector4<T>::operator/=(T f) {
  CHECK_FLOAT_EQUAL(f, 0.f);
  T inv = 1.f / f;
  x *= inv;
  y *= inv;
  z *= inv;
  w *= inv;
  return *this;
}

template <typename T> Vector4<T> Vector4<T>::operator-() const {
  return Vector4(-x, -y, -z, -w);
}

template <typename T> T Vector4<T>::length2() const {
  return x * x + y * y + z * z + w * w;
  ;
}

template <typename T> T Vector4<T>::length() const { return sqrtf(length2()); }

template <typename T> bool Vector4<T>::HasNaNs() const {
  return std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(w);
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Vector4<T> &v) {
  os << "[vector4]" << v.x << " " << v.y << " " << v.z << " " << v.w
     << std::endl;
  return os;
}

///////////////////////////////////////////////////////////////////////////////
template <typename T> Vector2<T> operator*(T f, const Vector2<T> &v) {
  return v * f;
}
template <typename T> Vector2<T> operator/(T f, const Vector2<T> &v) {
  return Vector2<T>(f / v.x, f / v.y);
}

template <typename T> T dot(const Vector2<T> &a, const Vector2<T> &b) {
  return a.x * b.x + a.y * b.y;
}

template <typename T> Vector2<T> normalize(const Vector2<T> &v) {
  return v / v.length();
}

template <typename T> Vector2<T> orthonormal(const Vector2<T> &v, bool first) {
  Vector2<T> n = normalize(v);
  if (first)
    return Vector2<T>(-n.y, n.x);
  return Vector2<T>(n.y, -n.x);
}

template <typename T>
Vector2<T> project(const Vector2<T> &a, const Vector2<T> &b) {
  return (dot(b, a) / b.length2()) * b;
}

template <typename T> T cross(const Vector2<T> &a, const Vector2<T> &b) {
  return a.x * b.y - a.y * b.x;
}

template <typename T> Vector3<T> operator*(T f, const Vector3<T> &v) {
  return v * f;
}

template <typename T> T dot(const Vector3<T> &a, const Vector3<T> &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T>
Vector3<T> cross(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>((a.y * b.z) - (a.z * b.y), (a.z * b.x) - (a.x * b.z),
                    (a.x * b.y) - (a.y * b.x));
}

template <typename T>
T triple(const Vector3<T> &a, const Vector3<T> &b, const Vector3<T> &c) {
  return dot(a, cross(b, c));
}

template <typename T> Vector3<T> normalize(const Vector3<T> &v) {
  if (v.length2() == 0.f)
    return v;
  return v / v.length();
}

template <typename T>
void tangential(const Vector3<T> &a, Vector3<T> &b, Vector3<T> &c) {
  b = normalize(cross(a, ((std::abs(a.y) > 0.f || std::abs(a.z) > 0.f)
                              ? Vector3<T>(1, 0, 0)
                              : Vector3<T>(0, 1, 1))));
  c = normalize(cross(a, b));
}

template <typename T> Vector3<T> cos(const Vector3<T> &v) {
  return Vector3<T>(std::cos(v.x), std::cos(v.y), std::cos(v.z));
}

template <typename T> Vector3<T> max(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

template <typename T> Vector3<T> abs(const Vector3<T> &a) {
  return Vector3<T>(std::abs(a.x), std::abs(a.y), std::abs(a.z));
}

template <typename T> Vector<int, 3> ceil(const Vector3<T> &v) {
  return Vector<int, 3>(static_cast<int>(v[0] + 0.5f),
                        static_cast<int>(v[1] + 0.5f),
                        static_cast<int>(v[2] + 0.5f));
}

template <typename T> Vector<int, 3> floor(const Vector3<T> &v) {
  return Vector<int, 3>(static_cast<int>(v[0]), static_cast<int>(v[1]),
                        static_cast<int>(v[2]));
}

template <typename T> Vector<int, 3> min(Vector<int, 3> a, Vector<int, 3> b) {
  return Vector<int, 3>(std::min(a[0], b[0]), std::min(a[1], b[1]),
                        std::min(a[2], b[2]));
}

template <typename T> Vector<int, 3> max(Vector<int, 3> a, Vector<int, 3> b) {
  return Vector<int, 3>(std::max(a[0], b[0]), std::max(a[1], b[1]),
                        std::max(a[2], b[2]));
}

template <typename T> Vector<int, 2> ceil(const Vector2<T> &v) {
  return Vector<int, 2>(static_cast<int>(v[0] + 0.5f),
                        static_cast<int>(v[1] + 0.5f));
}

template <typename T> Vector<int, 2> floor(const Vector2<T> &v) {
  return Vector<int, 2>(static_cast<int>(v[0]), static_cast<int>(v[1]));
}

template <typename T> Vector<int, 2> min(Vector<int, 2> a, Vector<int, 2> b) {
  return Vector<int, 2>(std::min(a[0], b[0]), std::min(a[1], b[1]));
}

template <typename T> Vector<int, 2> max(Vector<int, 2> a, Vector<int, 2> b) {
  return Vector<int, 2>(std::max(a[0], b[0]), std::max(a[1], b[1]));
}

template <typename T, size_t D> Vector<T, D>::Vector() {
  size = D;
  memset(v, 0, D * sizeof(T));
}

template <typename T, size_t D>
Vector<T, D>::Vector(std::initializer_list<T> values) : Vector() {
  int k = 0;
  for (auto value = values.begin(); value != values.end(); value++)
    v[k++] = *value;
}

template <typename T, size_t D>
Vector<T, D>::Vector(size_t n, const T *t) : Vector() {
  for (size_t i = 0; i < D && i < n; i++)
    v[i] = t[i];
}

template <typename T, size_t D> Vector<T, D>::Vector(const T &t) : Vector() {
  for (size_t i = 0; i < D; i++)
    v[i] = t;
}

template <typename T, size_t D>
Vector<T, D>::Vector(const T &x, const T &y) : Vector() {
  if (size > 1) {
    v[0] = x;
    v[1] = y;
  }
}

template <typename T, size_t D>
Vector<T, D>::Vector(const T &x, const T &y, const T &z) : Vector() {
  if (size > 2) {
    v[0] = x;
    v[1] = y;
    v[2] = z;
  }
}

template <typename T, size_t D> T Vector<T, D>::operator[](int i) const {
  ASSERT(i >= 0 && i <= static_cast<int>(size));
  return v[i];
}

template <typename T, size_t D> T &Vector<T, D>::operator[](int i) {
  ASSERT(i >= 0 && i <= static_cast<int>(size));
  return v[i];
}

template <typename T, size_t D>
bool Vector<T, D>::operator==(const Vector<T, D> &_v) const {
  for (size_t i = 0; i < size; i++)
    if (!IS_EQUAL(v[i], _v[i]))
      return false;
  return true;
}

template <typename T, size_t D>
bool Vector<T, D>::operator!=(const Vector<T, D> &_v) const {
  bool dif = false;
  for (size_t i = 0; i < size; i++)
    if (!IS_EQUAL(v[i], _v[i])) {
      dif = true;
      break;
    }
  return dif;
}

template <typename T, size_t D>
bool Vector<T, D>::operator<=(const Vector<T, D> &_v) const {
  for (size_t i = 0; i < size; i++)
    if (v[i] > _v[i])
      return false;
  return true;
}

template <typename T, size_t D>
bool Vector<T, D>::operator<(const Vector<T, D> &_v) const {
  for (size_t i = 0; i < size; i++)
    if (v[i] >= _v[i])
      return false;
  return true;
}

template <typename T, size_t D>
bool Vector<T, D>::operator>=(const Vector<T, D> &_v) const {
  for (size_t i = 0; i < size; i++)
    if (v[i] < _v[i])
      return false;
  return true;
}

template <typename T, size_t D>
bool Vector<T, D>::operator>(const Vector<T, D> &_v) const {
  for (size_t i = 0; i < size; i++)
    if (v[i] <= _v[i])
      return false;
  return true;
}

template <typename T, size_t D>
Vector<T, D> Vector<T, D>::operator-(const Vector<T, D> &_v) const {
  Vector<T, D> v_;
  for (size_t i = 0; i < D; i++)
    v_[i] = v[i] - _v[i];
  return v_;
}

template <typename T, size_t D>
Vector<T, D> Vector<T, D>::operator+(const Vector<T, D> &_v) const {
  Vector<T, D> v_;
  for (size_t i = 0; i < D; i++)
    v_[i] = v[i] + _v[i];
  return v_;
}

template <typename T, size_t D>
Vector<T, D> Vector<T, D>::operator*(const Vector<T, D> &_v) const {
  Vector<T, D> v_;
  for (size_t i = 0; i < D; i++)
    v_[i] = v[i] * _v[i];
  return v_;
}

template <typename T, size_t D>
Vector<T, D> Vector<T, D>::operator/(const Vector<T, D> &_v) const {
  Vector<T, D> v_;
  for (size_t i = 0; i < D; i++)
    v_[i] = v[i] / _v[i];
  return v_;
}

template <typename T, size_t D> Vector<T, D> Vector<T, D>::operator/=(T f) {
  for (size_t i = 0; i < D; i++)
    v[i] /= f;
  return *this;
}

template <typename T, size_t D>
Vector<T, D> Vector<T, D>::operator-=(const Vector<T, D> &_v) {
  for (size_t i = 0; i < D; i++)
    v[i] -= _v[i];
  return *this;
}

template <typename T, size_t D>
Vector<T, 2> Vector<T, D>::operator/(T f) const {
  T inv = static_cast<T>(1) / f;
  return Vector<T, 2>(v[0] * inv, v[1] * inv);
}

template <typename T, size_t D>
Vector<T, 2> Vector<T, D>::xy(size_t x, size_t y) const {
  return Vector<T, 2>(v[x], v[y]);
}

template <typename T, size_t D>
Vector<T, 2> Vector<T, D>::floatXY(size_t x, size_t y) const {
  return Vector<T, 2>(static_cast<float>(v[x]), static_cast<float>(v[y]));
}

template <typename T, size_t D>
Vector<float, 3> Vector<T, D>::floatXYZ(size_t x, size_t y, size_t z) {
  return Vector<float, 3>(static_cast<float>(v[x]), static_cast<float>(v[y]),
                          static_cast<float>(v[z]));
}

template <typename T, size_t D> T Vector<T, D>::max() const {
  T m = v[0];
  for (size_t i = 1; i < D; i++)
    m = std::max(m, v[i]);
  return m;
}

template <typename T, size_t D> T Vector<T, D>::length2() const {
  T sum = 0.f;
  for (size_t i = 0; i < size; i++)
    sum += SQR(v[i]);
  return sum;
}

template <typename T, size_t D> T Vector<T, D>::length() const {
  return std::sqrt(length2());
}

template <typename T, size_t D> Vector<T, D> Vector<T, D>::normalized() const {
  T d = length();
  Vector<T, D> r;
  for (size_t i = 0; i < size; i++)
    r[i] = v[i] / d;
  return r;
}

template <typename T, size_t D> Vector<T, 2> Vector<T, D>::right() const {
  return Vector<T, 2>(v[1], -v[0]);
}

template <typename T, size_t D> Vector<T, 2> Vector<T, D>::left() const {
  return Vector<T, 2>(-v[1], v[0]);
}
