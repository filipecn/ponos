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

template <typename T> ScalarGrid2D<T>::ScalarGrid2D() {
  this->useBorder = false;
}

template <typename T> ScalarGrid2D<T>::ScalarGrid2D(uint w, uint h) {
  this->useBorder = false;
  this->setDimensions(w, h);
  this->init();
}

template <typename T>
Vector<T, 2> ScalarGrid2D<T>::gradient(int i, int j) const {
  T left = (*this)(i - 1, j);
  T right = (*this)(i + 1, j);
  T down = (*this)(i, j - 1);
  T up = (*this)(i, j + 1);
  vec2 cs = this->cellSize();
  return 0.5f * Vector<T, 2>({right - left, up - down}) /
         Vector<T, 2>({cs.x, cs.y});
}

template <typename T>
Vector<T, 2> ScalarGrid2D<T>::gradient(float x, float y) const {
  Point<int, 2> c = (*this).cell(Point2(x, y));
  int i = c[0], j = c[1];
  Point2 gp = this->toGrid(Point2(x, y));
  return bilerp(gp.x - i, gp.y - j, gradient(i, j), gradient(i + 1, j),
                gradient(i + 1, j + 1), gradient(i, j + 1));
}

template <typename T> T ScalarGrid2D<T>::laplacian(float x, float y) const {
  return 0;
}

template <typename T> T ScalarGrid2D<T>::sample(float x, float y) const {
  Point2 gp = this->toGrid(Point2(x, y));
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
