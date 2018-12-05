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

template <typename T, size_t D> real_t Vector<T, D>::length2() const {
  real_t sum = 0.f;
  for (size_t i = 0; i < size; i++)
    sum += SQR(v[i]);
  return sum;
}

template <typename T, size_t D> real_t Vector<T, D>::length() const {
  return std::sqrt(length2());
}

template <typename T, size_t D> Vector<T, D> Vector<T, D>::normalized() const {
  real_t d = length();
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
