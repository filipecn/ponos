template <typename T> Point2<T>::Point2() { x = y = 0.f; }

template <typename T> Point2<T>::Point2(real_t f) { x = y = f; }

template <typename T> Point2<T>::Point2(const real_t *v) : x(v[0]), y(v[1]) {
  ASSERT(!HasNaNs());
}

template <typename T> Point2<T>::Point2(real_t _x, real_t _y) : x(_x), y(_y) {
  ASSERT(!HasNaNs());
}

template <typename T> real_t Point2<T>::operator[](int i) const {
  ASSERT(i >= 0 && i <= 1);
  return (&x)[i];
}

template <typename T> real_t &Point2<T>::operator[](int i) {
  ASSERT(i >= 0 && i <= 1);
  return (&x)[i];
}

template <typename T> bool Point2<T>::operator==(const Point2 &p) const {
  return IS_EQUAL(x, p.x) && IS_EQUAL(y, p.y);
}

template <typename T> Point2<T> Point2<T>::operator+(const Vector2<T> &v) const {
  return Point2(x + v.x, y + v.y);
}

template <typename T> Point2<T> Point2<T>::operator-(const Vector2<T> &v) const {
  return Point2(x - v.x, y - v.y);
}

template <typename T> Point2<T> Point2<T>::operator-(const real_t &f) const {
  return Point2(x - f, y - f);
}

template <typename T> Point2<T> Point2<T>::operator+(const real_t &f) const {
  return Point2(x + f, y + f);
}

template <typename T> Vector2<T> Point2<T>::operator-(const Point2 &p) const {
  return Vector2<T>(x - p.x, y - p.y);
};

template <typename T> Point2<T> Point2<T>::operator/(real_t d) const {
  return Point2(x / d, y / d);
}

template <typename T> Point2<T> Point2<T>::operator*(real_t f) const {
  return Point2(x * f, y * f);
}

template <typename T> Point2<T> &Point2<T>::operator+=(const Vector2<T> &v) {
  x += v.x;
  y += v.y;
  return *this;
}

template <typename T> Point2<T> &Point2<T>::operator-=(const Vector2<T> &v) {
  x -= v.x;
  y -= v.y;
  return *this;
}

template <typename T> Point2<T> &Point2<T>::operator/=(real_t d) {
  x /= d;
  y /= d;
  return *this;
}

template <typename T> bool Point2<T>::operator<(const Point2 &p) const {
  if (x >= p.x || y >= p.y)
    return false;
  return true;
}

template <typename T> bool Point2<T>::operator>=(const Point2 &p) const {
  return x >= p.x && y >= p.y;
}

template <typename T> bool Point2<T>::operator<=(const Point2 &p) const {
  return x <= p.x && y <= p.y;
}

template <typename T> bool Point2<T>::HasNaNs() const {
  return std::isnan(x) || std::isnan(y);
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Point2<T> &p) {
  os << "[Point2] " << p.x << " " << p.y << std::endl;
  return os;
}

template <typename T> Point3<T>::Point3() { x = y = z = 0.0f; }

template <typename T>
Point3<T>::Point3(real_t _x, real_t _y, real_t _z) : x(_x), y(_y), z(_z) {
  ASSERT(!HasNaNs());
}

template <typename T>
Point3<T>::Point3(const Vector3<T> &v) : x(v.x), y(v.y), z(v.z) {
  ASSERT(!HasNaNs());
}

template <typename T>
Point3<T>::Point3(const Point2<T> &p) : x(p.x), y(p.y), z(0) {}

template <typename T>
Point3<T>::Point3(const real_t *v) : x(v[0]), y(v[1]), z(v[2]) {
  ASSERT(!HasNaNs());
}

template <typename T> real_t Point3<T>::operator[](int i) const {
  ASSERT(i >= 0 && i <= 2);
  return (&x)[i];
}

template <typename T> real_t &Point3<T>::operator[](int i) {
  ASSERT(i >= 0 && i <= 2);
  return (&x)[i];
}

// arithmetic
template <typename T> Point3<T> Point3<T>::operator+(const Vector3<T> &v) const {
  return Point3(x + v.x, y + v.y, z + v.z);
}

template <typename T> Point3<T> Point3<T>::operator+(const real_t &f) const {
  return Point3(x + f, y + f, z + f);
}

template <typename T> Point3<T> Point3<T>::operator-(const real_t &f) const {
  return Point3(x - f, y - f, z - f);
}

template <typename T> Point3<T> &Point3<T>::operator+=(const Vector3<T> &v) {
  x += v.x;
  y += v.y;
  z += v.z;
  return *this;
}

template <typename T> Vector3<T> Point3<T>::operator-(const Point3 &p) const {
  return Vector3<T>(x - p.x, y - p.y, z - p.z);
}

template <typename T> Point3<T> Point3<T>::operator-(const Vector3<T> &v) const {
  return Point3(x - v.x, y - v.y, z - v.z);
}

template <typename T> Point3<T> &Point3<T>::operator-=(const Vector3<T> &v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;
  return *this;
}

template <typename T> bool Point3<T>::operator==(const Point3 &p) const {
  return IS_EQUAL(p.x, x) && IS_EQUAL(p.y, y) && IS_EQUAL(p.z, z);
}

template <typename T> bool Point3<T>::operator>=(const Point3 &p) const {
  return x >= p.x && y >= p.y && z >= p.z;
}

template <typename T> bool Point3<T>::operator<=(const Point3 &p) const {
  return x <= p.x && y <= p.y && z <= p.z;
}

template <typename T> Point3<T> Point3<T>::operator*(real_t d) const {
  return Point3(x * d, y * d, z * d);
}

template <typename T> Point3<T> Point3<T>::operator/(real_t d) const {
  return Point3(x / d, y / d, z / d);
}

template <typename T> Point3<T> &Point3<T>::operator/=(real_t d) {
  x /= d;
  y /= d;
  z /= d;
  return *this;
}

template <typename T> bool Point3<T>::operator==(const Point3 &p) {
  return IS_EQUAL(x, p.x) && IS_EQUAL(y, p.y) && IS_EQUAL(z, p.z);
}

template <typename T> Point2<T> Point3<T>::xy() const { return Point2<T>(x, y); }

template <typename T> Point2<T> Point3<T>::yz() const { return Point2<T>(y, z); }

template <typename T> Point2<T> Point3<T>::xz() const { return Point2<T>(x, z); }

template <typename T> Vector3<T> Point3<T>::asVector3() const {
  return Vector3<T>(x, y, z);
}

template <typename T> ivec3 Point3<T>::asIVec3() const {
  return ivec3(static_cast<const int &>(x), static_cast<const int &>(y),
               static_cast<const int &>(z));
}

template <typename T> bool Point3<T>::HasNaNs() const {
  return std::isnan(x) || std::isnan(y) || std::isnan(z);
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Point3<T> &p) {
  os << "[Point3] " << p.x << " " << p.y << " " << p.z << std::endl;
  return os;
}

template<typename T>
Point3<T> &Point3<T>::operator*=(T d) {
  x *= d;
  y *= d;
  z *= d;
  return *this;
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline Point2<T> operator*(real_t f, const Point2<T> &p) { return p * f; }

template <typename T>
inline real_t distance(const Point2<T> &a, const Point2<T> &b) {
  return (a - b).length();
}

template <typename T>
inline real_t distance2(const Point2<T> &a, const Point2<T> &b) {
  return (a - b).length2();
}

template <typename T>
inline real_t distance(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length();
}

template <typename T>
inline real_t distance2(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length2();
}

template <class T, size_t D> Point<T, D>::Point() {
  size = D;
  for (size_t i = 0; i < D; i++)
    v[i] = static_cast<T>(0);
}

template <class T, size_t D> Point<T, D>::Point(T v) {
  size = D;
  for (size_t i = 0; i < D; i++)
    v[i] = static_cast<T>(v);
}

template <class T, size_t D> Point<T, D>::Point(Point2<T> p) {
  size = D;
  v[0] = static_cast<T>(p.x);
  v[1] = static_cast<T>(p.y);
}

template <class T, size_t D>
inline bool operator==(const Point<T, D> &lhs, const Point<T, D> &rhs) {
  for (size_t i = 0; i < lhs.size; ++i)
    if (lhs[i] != rhs[i])
      return false;
  return true;
}

template <class T, size_t D>
inline bool operator!=(const Point<T, D> &lhs, const Point<T, D> &rhs) {
  return !(lhs == rhs);
}

template <typename T, size_t D> Point<T, D>::Point(std::initializer_list<T> p) {
  size = D;
  int k = 0;
  for (auto it = p.begin(); it != p.end(); ++it) {
    if (k >= D)
      break;
    v[k++] = *it;
  }
}

template <typename T, size_t D> T Point<T, D>::operator[](int i) const {
  ASSERT(i >= 0 && i <= static_cast<int>(size));
  return v[i];
}

template <typename T, size_t D> T &Point<T, D>::operator[](int i) {
  ASSERT(i >= 0 && i <= static_cast<int>(size));
  return v[i];
}

template <typename T, size_t D>
bool Point<T, D>::operator>=(const Point<T, D> &p) const {
  for (int i = 0; i < D; i++)
    if (v[i] < p[i])
      return false;
  return true;
}

template <typename T, size_t D>
bool Point<T, D>::operator<=(const Point<T, D> &p) const {
  for (int i = 0; i < D; i++)
    if (v[i] > p[i])
      return false;
  return true;
}

template <typename T, size_t D>
Vector<T, D> Point<T, D>::operator-(const Point<T, D> &p) const {
  Vector<T, D> V;
  for (int i = 0; i < D; i++)
    V[i] = v[i] - p[i];
  return V;
}

template <typename T, size_t D>
Point<T, D> Point<T, D>::operator+(const Vector<T, D> &V) const {
  Point<T, D> P;
  for (int i = 0; i < D; i++)
    P[i] = v[i] + V[i];
  return P;
}

template <typename T, size_t D>
Point2<T> Point<T, D>::floatXY(size_t x, size_t y) const {
  return Point2<T>(static_cast<float>(v[x]), static_cast<float>(v[y]));
}
