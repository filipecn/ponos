template <typename T> BBox2<T>::BBox2() {
  lower = Point2<T>(Constants::greatest<T>());
  upper = Point2<T>(Constants::lowest<T>());
}

template <typename T>
BBox2<T>::BBox2(const Point2<T> &p) : lower(p), upper(p) {}

template <typename T>
BBox2<T>::BBox2(const Point2<T> &p1, const Point2<T> &p2) {
  lower = Point2<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y));
  upper = Point2<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
}

template <typename T> BBox2<T> BBox2<T>::unitBox() {
  return {Point2<T>(), Point2<T>(1, 1)};
}

template <typename T> bool BBox2<T>::inside(const Point2<T> &p) const {
  return (p.x >= lower.x && p.x <= upper.x && p.y >= lower.y && p.y <= upper.y);
}

template <typename T> real_t BBox2<T>::size(int d) const {
  d = std::max(0, std::min(1, d));
  return upper[d] - lower[d];
}

template <typename T> Vector2<T> BBox2<T>::extends() const {
  return upper - lower;
}

template <typename T> Point2<T> BBox2<T>::center() const {
  return lower + (upper - lower) * .5f;
}

template <typename T> Point2<T> BBox2<T>::centroid() const {
  return lower * .5f + vec2(upper * .5f);
}

template <typename T> int BBox2<T>::maxExtent() const {
  Vector2<T> diag = upper - lower;
  if (diag.x > diag.y)
    return 0;
  return 1;
}

template <typename T> const Point2<T> &BBox2<T>::operator[](int i) const {
  return (i == 0) ? lower : upper;
}

template <typename T> Point2<T> &BBox2<T>::operator[](int i) {
  return (i == 0) ? lower : upper;
}

template <typename T> BBox3<T>::BBox3() {
  lower = Point3<T>(Constants::greatest<T>());
  upper = Point3<T>(Constants::lowest<T>());
}

template <typename T>
BBox3<T>::BBox3(const Point3<T> &p) : lower(p), upper(p) {}

template <typename T>
BBox3<T>::BBox3(const Point3<T> &p1, const Point3<T> &p2) {
  lower = Point3<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
                    std::min(p1.z, p2.z));
  upper = Point3<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
                    std::max(p1.z, p2.z));
}

template <typename T> BBox3<T>::BBox3(const Point3<T> &c, float r) {
  lower = c - Vector3<T>(r, r, r);
  upper = c + Vector3<T>(r, r, r);
}

template <typename T> BBox3<T> BBox3<T>::unitBox() {
  return {Point3<T>(), Point3<T>(1, 1, 1)};
}

template <typename T> bool BBox3<T>::contains(const Point3<T> &p) const {
  return (p.x >= lower.x && p.x <= upper.x && p.y >= lower.y &&
          p.y <= upper.y && p.z >= lower.z && p.z <= upper.z);
}

template <typename T> bool BBox3<T>::contains(const BBox3 &b) const {
  return contains(b.lower) && contains(b.upper);
}

template <typename T>
bool BBox3<T>::containsExclusive(const Point3<T> &p) const {
  return (p.x >= lower.x && p.x < upper.x && p.y >= lower.y && p.y < upper.y &&
          p.z >= lower.z && p.z < upper.z);
}

template <typename T> void BBox3<T>::expand(real_t delta) {
  lower -= Vector3<T>(delta, delta, delta);
  upper += Vector3<T>(delta, delta, delta);
}

template <typename T> Vector3<T> BBox3<T>::diagonal() const {
  return upper - lower;
}

template <typename T> std::vector<BBox3<T>> BBox3<T>::splitBy8() const {
  auto mid = center();
  std::vector<BBox3<T>> children;
  children.emplace_back(lower, mid);
  children.emplace_back(Point3<T>(mid.x, lower.y, lower.z),
                        Point3<T>(upper.x, mid.y, mid.z));
  children.emplace_back(Point3<T>(lower.x, mid.y, lower.z),
                        Point3<T>(mid.x, upper.y, mid.z));
  children.emplace_back(Point3<T>(mid.x, mid.y, lower.z),
                        Point3<T>(upper.x, upper.y, mid.z));
  children.emplace_back(Point3<T>(lower.x, lower.y, mid.z),
                        Point3<T>(mid.x, mid.y, upper.z));
  children.emplace_back(Point3<T>(mid.x, lower.y, mid.z),
                        Point3<T>(upper.x, mid.y, upper.z));
  children.emplace_back(Point3<T>(lower.x, mid.y, mid.z),
                        Point3<T>(mid.x, upper.y, upper.z));
  children.emplace_back(Point3<T>(mid.x, mid.y, mid.z),
                        Point3<T>(upper.x, upper.y, upper.z));
  return children;
}

template <typename T> Point3<T> BBox3<T>::center() const {
  return lower + (upper - lower) * .5f;
}

template <typename T> T BBox3<T>::size(size_t d) const {
  return upper[d] - lower[d];
}

template <typename T> int BBox3<T>::maxExtent() const {
  Vector3<T> diag = upper - lower;
  if (diag.x > diag.y && diag.x > diag.z)
    return 0;
  else if (diag.y > diag.z)
    return 1;
  return 2;
}

template <typename T> BBox2<T> BBox3<T>::xy() const {
  return BBox2<T>(lower.xy(), upper.xy());
}

template <typename T> BBox2<T> BBox3<T>::yz() const {
  return BBox2<T>(lower.yz(), upper.yz());
}

template <typename T> BBox2<T> BBox3<T>::xz() const {
  return BBox2<T>(lower.xz(), upper.xz());
}

template <typename T> Point3<T> BBox3<T>::centroid() const {
  return lower * .5f + vec3(upper * .5f);
}

template <typename T> const Point3<T> &BBox3<T>::operator[](int i) const {
  return (i == 0) ? lower : upper;
}

template <typename T> Point3<T> &BBox3<T>::operator[](int i) {
  return (i == 0) ? lower : upper;
}

template <typename T> Point3<T> BBox3<T>::corner(int c) const {
  return Point3<T>((*this)[(c & 1)].x, (*this)[(c & 2) ? 1 : 0].y,
                   (*this)[(c & 4) ? 1 : 0].z);
}

template <typename T>
BBox3<T> make_union(const BBox3<T> &b, const Point3<T> &p) {
  BBox3<T> ret = b;
  ret.lower.x = std::min(b.lower.x, p.x);
  ret.lower.y = std::min(b.lower.y, p.y);
  ret.lower.z = std::min(b.lower.z, p.z);
  ret.upper.x = std::max(b.upper.x, p.x);
  ret.upper.y = std::max(b.upper.y, p.y);
  ret.upper.z = std::max(b.upper.z, p.z);
  return ret;
}

template <typename T>
BBox3<T> make_union(const BBox3<T> &a, const BBox3<T> &b) {
  BBox3<T> ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}

template <typename T> bool overlaps(const BBox3<T> &a, const BBox3<T> &b) {
  bool x = (a.upper.x >= b.lower.x) && (a.lower.x <= b.upper.x);
  bool y = (a.upper.y >= b.lower.y) && (a.lower.y <= b.upper.y);
  bool z = (a.upper.z >= b.lower.z) && (a.lower.z <= b.upper.z);
  return (x && y && z);
}

template <typename T> BBox3<T> intersect(const BBox3<T> &a, const BBox3<T> &b) {
  return BBox3<T>(
      Point3<T>(std::max(a.lower.x, b.lower.x), std::max(a.lower.x, b.lower.y),
                std::max(a.lower.z, b.lower.z)),
      Point3<T>(std::min(a.lower.x, b.lower.x), std::min(a.lower.x, b.lower.y),
                std::min(a.lower.z, b.lower.z)));
}
