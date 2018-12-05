template<class T, size_t D>
Point<T, D>::Point() {
  size = D;
  for (size_t i = 0; i < D; i++)
    v[i] = static_cast<T>(0);
}

template<class T, size_t D>
Point<T, D>::Point(T v) {
  size = D;
  for (size_t i = 0; i < D; i++)
    v[i] = static_cast<T>(v);
}

template<class T, size_t D>
Point<T, D>::Point(Point2 p) {
  size = D;
  v[0] = static_cast<T>(p.x);
  v[1] = static_cast<T>(p.y);
}

template<class T, size_t D>
inline bool operator==(const Point <T, D> &lhs, const Point <T, D> &rhs) {
  for (size_t i = 0; i < lhs.size; ++i)
    if (lhs[i] != rhs[i])
      return false;
  return true;
}

template<class T, size_t D>
inline bool operator!=(const Point <T, D> &lhs, const Point <T, D> &rhs) {
  return !(lhs == rhs);
}

template<typename T, size_t D>
Point<T, D>::Point(std::initializer_list<T> p) {
  size = D;
  int k = 0;
  for (auto it = p.begin(); it != p.end(); ++it) {
    if (k >= D)
      break;
    v[k++] = *it;
  }
}

template<typename T, size_t D>
T Point<T, D>::operator[](int i) const {
  ASSERT(i >= 0 && i <= static_cast<int>(size));
  return v[i];
}

template<typename T, size_t D>
T &Point<T, D>::operator[](int i) {
  ASSERT(i >= 0 && i <= static_cast<int>(size));
  return v[i];
}

template<typename T, size_t D>
bool Point<T, D>::operator>=(const Point <T, D> &p) const {
  for (int i = 0; i < D; i++)
    if (v[i] < p[i])
      return false;
  return true;
}

template<typename T, size_t D>
bool Point<T, D>::operator<=(const Point <T, D> &p) const {
  for (int i = 0; i < D; i++)
    if (v[i] > p[i])
      return false;
  return true;
}

template<typename T, size_t D>
Vector <T, D> Point<T, D>::operator-(const Point <T, D> &p) const {
  Vector<T, D> V;
  for (int i = 0; i < D; i++)
    V[i] = v[i] - p[i];
  return V;
}

template<typename T, size_t D>
Point <T, D> Point<T, D>::operator+(const Vector <T, D> &V) const {
  Point<T, D> P;
  for (int i = 0; i < D; i++)
    P[i] = v[i] + V[i];
  return P;
}

template<typename T, size_t D>
Point2 Point<T, D>::floatXY(size_t x, size_t y) const {
  return Point2(static_cast<float>(v[x]), static_cast<float>(v[y]));
}
