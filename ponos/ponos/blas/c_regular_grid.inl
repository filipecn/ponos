/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

template<typename T>
CRegularGrid<T>::CRegularGrid(const size3 &d, const T &b, const vec3 cellSize,
                              const vec3 &offset) {
  this->dimensions = d;
  this->background = b;
  data = new T **[d[0]];
  for (int i = 0; i < static_cast<i64>(d[0]); i++)
    data[i] = new T *[d[1]];
  for (int i = 0; i < static_cast<i64>(d[0]); i++)
    for (int j = 0; j < static_cast<i64>(d[1]); j++)
      data[i][j] = new T[d[2]];
  this->toWorld.reset();
  this->toWorld.scale(cellSize.x, cellSize.y, cellSize.z);
  this->toWorld.translate(offset);
  this->toWorld.computeInverse();
  this->toGrid = inverse(this->toWorld);
}

template<typename T>
CRegularGrid<T>::CRegularGrid(const size3 &d, const T &b, const bbox3 &bb) {
  this->dimensions = d;
  this->background = b;
  data = new T **[d[0]];
  for (int i = 0; i < d[0]; i++)
    data[i] = new T *[d[1]];
  for (int i = 0; i < d[0]; i++)
    for (int j = 0; j < d[1]; j++)
      data[i][j] = new T[d[2]];
  this->toWorld.reset();
  vec3 s = vec3(bb.size(0) / (d[0]), bb.size(1) / (d[1]), bb.size(2) / (d[2]));
  this->toWorld =
      translate(vec3(bb.lower) + s * 0.5f) * scale(s[0], s[1], s[2]);

  // this->toWorld.scale(s[0], s[1], s[2]);
  // this->toWorld.translate(vec3(bb.lower) + s * 0.5f + vec3(0, 0, 1));
  this->toWorld.computeInverse();
  this->toGrid = inverse(this->toWorld);
}

template<typename T> CRegularGrid<T>::~CRegularGrid() {
  for (u32 i = 0; i < this->dimensions[0]; i++)
    for (u32 j = 0; j < this->dimensions[1]; j++)
      delete[] data[i][j];
  for (u32 i = 0; i < this->dimensions[0]; i++)
    delete[] data[i];
  delete[] data;
}

template<typename T> void CRegularGrid<T>::setAll(T v) {
  for (auto ijk : Index3Range<i32>(this->dimensions))
    data[ijk[0]][ijk[1]][ijk[2]] = v;
}

template<typename T> T CRegularGrid<T>::safeData(int i, int j, int k) const {
  return data[max(0, min(this->dimensions[0] - 1, i))]
  [max(0, min(this->dimensions[1] - 1, j))]
  [max(0, min(this->dimensions[2] - 1, k))];
}

template<typename T> void CRegularGrid<T>::set(const size3 &i, const T &v) {
  this->data[std::max(0u, std::min(this->dimensions[0] - 1, i[0]))]
  [std::max(0u, std::min(this->dimensions[1] - 1, i[1]))]
  [std::max(0u, std::min(this->dimensions[2] - 1, i[2]))] = v;
}

template<typename T> void CRegularGrid<T>::normalize() {
  T M = data[0][0][0];
  for (auto ijk : Index3Range<i32>(this->dimensions))
    M = ponos::max(M, data[ijk[0]][ijk[1]][ijk[2]]);
  for (auto ijk : Index3Range<i32>(this->dimensions))
    data[ijk[0]][ijk[1]][ijk[2]] /= M;
}

template<typename T> void CRegularGrid<T>::normalizeElements() {
  for (auto ijk : Index3Range<i32>(this->dimensions))
    data[ijk[0]][ijk[1]][ijk[2]] =
        ponos::normalize(data[ijk[0]][ijk[1]][ijk[2]]);
}

template<typename T> CRegularGrid2D<T>::CRegularGrid2D() {}

template<typename T>
CRegularGrid2D<T>::CRegularGrid2D(uint32_t w, uint32_t h) {
  this->setDimensions(w, h);
}

template<typename T> T CRegularGrid2D<T>::sample(float x, float y) const {
  point2 gp = this->toGrid(point2(x, y));
  int x0 = static_cast<int>(gp.x);
  int y0 = static_cast<int>(gp.y);
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  x0 = std::max(0, std::min(static_cast<int>(this->width) - 1, x0));
  y0 = std::max(0, std::min(static_cast<int>(this->height) - 1, y0));
  x1 = std::max(0, std::min(static_cast<int>(this->width) - 1, x1));
  y1 = std::max(0, std::min(static_cast<int>(this->height) - 1, y1));

  T p[4][4];
  int delta[] = {-1, 0, 1, 2};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      p[i][j] = this->safeData(x0 + delta[i], y0 + delta[j]);
  return ponos::bicubicInterpolate<T>(p, gp.x - x0, gp.y - y0);
}

template<typename T> void CRegularGrid2D<T>::updateDataStructure() {
  data.resize(this->width, std::vector<T>());
  for (size_t i = 0; i < this->width; i++)
    data[i].resize(this->height);
}
