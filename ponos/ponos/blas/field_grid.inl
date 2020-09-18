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

template <typename T> ScalarGrid2D<T>::ScalarGrid2D() {}

template <typename T> ScalarGrid2D<T>::ScalarGrid2D(uint w, uint h) {
  this->dataPosition = GridDataPosition::CELL_CENTER;
  this->accessMode = GridAccessMode::CLAMP_TO_EDGE;
  this->setDimensions(w, h);
}

template <typename T>
ScalarGrid2D<T>::ScalarGrid2D(uint w, uint h, const bbox2 &b) {
  this->dataPosition = GridDataPosition::CELL_CENTER;
  this->accessMode = GridAccessMode::CLAMP_TO_EDGE;
  this->set(w, h, b);
}

template <typename T> Vector2<T> ScalarGrid2D<T>::gradient(int i, int j) const {
  T left = (*this)(i - 1, j);
  T right = (*this)(i + 1, j);
  T down = (*this)(i, j - 1);
  T up = (*this)(i, j + 1);
  vec2 cs = this->cellSize();
  return 0.5f * Vector2<T>({right - left, up - down}) / Vector2<T>(cs.x, cs.y);
}

template <typename T>
Vector2<T> ScalarGrid2D<T>::gradient(float x, float y) const {
  index2 c = (*this).dataCell(point2(x, y));
  int i = c[0], j = c[1];
  point2 gp = this->dataGridPosition(point2(x, y));
  return bilerp(gp.x - i, gp.y - j, gradient(i, j), gradient(i + 1, j),
                gradient(i + 1, j + 1), gradient(i, j + 1));
}

template <typename T> T ScalarGrid2D<T>::laplacian(float x, float y) const {
  UNUSED_VARIABLE(x);
  UNUSED_VARIABLE(y);
  return 0;
}

template <typename T> T ScalarGrid2D<T>::sample(float x, float y) const {
  /*Point<int, 2> c = (*this).dataCell(point2(x, y));
  int i = c[0], j = c[1];
  point2 gp = this->dataGridPosition(point2(x, y));
  return bilerp(gp.x - i, gp.y - j, (*this)(i, j), (*this)(i + 1, j),
                (*this)(i + 1, j + 1), (*this)(i, j + 1));
  */
  point2 gp = this->dataGridPosition(point2(x, y));
  int x0 = static_cast<int>(gp.x);
  int y0 = static_cast<int>(gp.y);
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  x0 = std::max(0, std::min(static_cast<int>(this->width) - 1, x0));
  y0 = std::max(0, std::min(static_cast<int>(this->height) - 1, y0));
  x1 = std::max(0, std::min(static_cast<int>(this->width) - 1, x1));
  y1 = std::max(0, std::min(static_cast<int>(this->height) - 1, y1));

  float p[4][4];
  int delta[] = {-1, 0, 1, 2};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      p[i][j] = this->safeData(x0 + delta[i], y0 + delta[j]);
  return ponos::bicubicInterpolate<float>(p, gp.x - x0, gp.y - y0);
}

template <typename T> VectorGrid2D<T>::VectorGrid2D() {}

template <typename T> VectorGrid2D<T>::VectorGrid2D(uint w, uint h) {
  this->dataPosition = GridDataPosition::CELL_CENTER;
  this->accessMode = GridAccessMode::CLAMP_TO_EDGE;
  this->setDimensions(w, h);
}

template <typename T>
VectorGrid2D<T>::VectorGrid2D(uint w, uint h, const bbox2 &b) {
  this->dataPosition = GridDataPosition::CELL_CENTER;
  this->accessMode = GridAccessMode::CLAMP_TO_EDGE;
  this->set(w, h, b);
}

template <typename T> T VectorGrid2D<T>::divergence(int i, int j) const {
  vec2 cs = 2 * this->cellSize();
  return ((*this)(i + 1, j)[0] - (*this)(i - 1, j)[0]) / cs[0] +
         ((*this)(i, j + 1)[1] - (*this)(i, j - 1)[1]) / cs[1];
}

template <typename T> T VectorGrid2D<T>::divergence(float x, float y) const {
  return static_cast<T>(0.0);
}

template <typename T> Vector2<T> VectorGrid2D<T>::curl(float x, float y) const {
  return Vector2<T>();
}

template <typename T>
Vector2<T> VectorGrid2D<T>::sample(float x, float y) const {
  return Vector2<T>();
}

template <typename T>
void computeDivergenceField(const VectorGrid2D<T> &vectorGrid,
                            ScalarGrid2D<T> *scalarGrid) {
  size2 d = vectorGrid.getDimensions();
  if (scalarGrid->getDimensions() != d)
    scalarGrid->setDimensions(d[0], d[1]);
  scalarGrid->setTransform(vectorGrid.toWorld);
  scalarGrid->accessMode = vectorGrid.accessMode;
  scalarGrid->dataPosition = vectorGrid.dataPosition;
  scalarGrid->forEach([&vectorGrid](T &d, size_t i, size_t j) {
    d = vectorGrid.divergence(static_cast<int>(i), static_cast<int>(j));
  });
}
