template <class T> Interval<T>::Interval() : low(0), high(0) {}

template <class T> Interval<T>::Interval(T l, T h) : low(l), high(h) {}

template <class T> Interval<T> Interval<T>::operator+(const Interval &i) const {
  return Interval(low + i.low, high + i.high);
}

template <class T> Interval<T> Interval<T>::operator-(const Interval &i) const {
  return Interval(low - i.high, high - i.low);
}

template <class T> Interval<T> Interval<T>::operator*(const Interval &i) const {
  Interval r;
  T prod[4] = {low * i.low, high * i.low, low * i.high, high * i.high};
  r.low = std::min(std::min(prod[0], prod[1]), std::min(prod[2], prod[3]));
  r.high = std::max(std::max(prod[0], prod[1]), std::max(prod[2], prod[3]));
  return r;
}

template <class T> Interval<T> Interval<T>::operator/(const Interval &i) const {
  Interval r = i;
  if (r.low < 0 && r.high > 0) {
    r.low = -Constants::lowest<T>();
    r.high = Constants::greatest<T>();
  } else {
    T div[4] = {low / r.low, high / r.low, low / r.high, high / r.high};
    r.low = std::min(std::min(div[0], div[1]), std::min(div[2], div[3]));
    r.high = std::max(std::max(div[0], div[1]), std::max(div[2], div[3]));
  }
  return r;
}
