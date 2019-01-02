template <int nSpectrumSamples>
real_t &CoefficientSpectrum<nSpectrumSamples>::operator[](int i) {
  return c[i];
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples>::CoefficientSpectrum(real_t v) {
  for (int i = 0; i < nSpectrumSamples; i++)
    c[i] = v;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> &CoefficientSpectrum<nSpectrumSamples>::
operator+=(const real_t &f) {
  for (int i = 0; i < nSpectrumSamples; i++)
    c[i] += f;
  return *this;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> CoefficientSpectrum<nSpectrumSamples>::
operator+(const real_t &f) const {
  CoefficientSpectrum<nSpectrumSamples> r = *this;
  for (int i = 0; i < nSpectrumSamples; i++)
    r.c[i] += f;
  return r;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> &CoefficientSpectrum<nSpectrumSamples>::
operator+=(const CoefficientSpectrum<nSpectrumSamples> &s2) {
  for (int i = 0; i < nSpectrumSamples; i++)
    c[i] += s2.c[i];
  return *this;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> CoefficientSpectrum<nSpectrumSamples>::
operator+(const CoefficientSpectrum<nSpectrumSamples> &s2) const {
  CoefficientSpectrum<nSpectrumSamples> r = *this;
  for (int i = 0; i < nSpectrumSamples; i++)
    r.c[i] += s2.c[i];
  return r;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> CoefficientSpectrum<nSpectrumSamples>::
operator-(const CoefficientSpectrum<nSpectrumSamples> &s2) const {
  CoefficientSpectrum<nSpectrumSamples> r = *this;
  for (int i = 0; i < nSpectrumSamples; i++)
    r.c[i] -= s2.c[i];
  return r;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> &CoefficientSpectrum<nSpectrumSamples>::
operator*=(const CoefficientSpectrum<nSpectrumSamples> &s2) {
  for (int i = 0; i < nSpectrumSamples; i++)
    c[i] *= s2.c[i];
  return *this;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> CoefficientSpectrum<nSpectrumSamples>::
operator*(const CoefficientSpectrum<nSpectrumSamples> &s2) const {
  CoefficientSpectrum<nSpectrumSamples> r = *this;
  for (int i = 0; i < nSpectrumSamples; i++)
    r.c[i] *= s2.c[i];
  return r;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> &CoefficientSpectrum<nSpectrumSamples>::
operator*=(const real_t &f) {
  for (int i = 0; i < nSpectrumSamples; i++)
    c[i] *= f;
  return *this;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> CoefficientSpectrum<nSpectrumSamples>::
operator*(const real_t &f) const {
  CoefficientSpectrum<nSpectrumSamples> r = *this;
  for (int i = 0; i < nSpectrumSamples; i++)
    r.c[i] *= f;
  return r;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> &CoefficientSpectrum<nSpectrumSamples>::
operator/=(const CoefficientSpectrum<nSpectrumSamples> &s2) {
  for (int i = 0; i < nSpectrumSamples; i++)
    c[i] /= s2.c[i];
  return *this;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> CoefficientSpectrum<nSpectrumSamples>::
operator/(const CoefficientSpectrum<nSpectrumSamples> &s2) const {
  CoefficientSpectrum<nSpectrumSamples> r = *this;
  for (int i = 0; i < nSpectrumSamples; i++)
    r.c[i] /= s2.c[i];
  return r;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> &CoefficientSpectrum<nSpectrumSamples>::
operator/=(const real_t &f) {
  for (int i = 0; i < nSpectrumSamples; i++)
    c[i] /= f;
  return *this;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples> CoefficientSpectrum<nSpectrumSamples>::
operator/(const real_t &f) const {
  CoefficientSpectrum<nSpectrumSamples> r = *this;
  for (int i = 0; i < nSpectrumSamples; i++)
    r.c[i] /= f;
  return r;
}

template <int nSpectrumSamples>
bool CoefficientSpectrum<nSpectrumSamples>::isBlack() const {
  for (int i = 0; i < nSpectrumSamples; i++)
    if (c[i] != 0.)
      return false;
  return true;
}

template <int nSpectrumSamples>
bool CoefficientSpectrum<nSpectrumSamples>::hasNaNs() const {
  for (int i = 0; i < nSpectrumSamples; i++)
    if (std::isnan(c[i]))
      return true;
  return false;
}

template <int nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples>
CoefficientSpectrum<nSpectrumSamples>::clamp(real_t l, real_t h) const {
  CoefficientSpectrum r;
  for (int i = 0; i < nSpectrumSamples; i++)
    r.c[i] = ponos::clamp(c[i], l, h);
  return r;
}
