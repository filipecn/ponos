
///////////////////////////////////////////////////////////////////////////////

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
