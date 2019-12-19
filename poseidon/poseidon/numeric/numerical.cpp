// Created by filipecn on 2/27/18.
/*
 * Copyright (c) 2018 FilipeCN
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

#include <poseidon/math/numerical.h>

namespace poseidon {

ponos::vec3d enrightField(ponos::Point3 p) {
  return ponos::vec3d({2. * std::sin(ponos::Constants::pi * p.x) * std::sin(2. * ponos::Constants::pi * p.y)
                           * std::sin(2. * ponos::Constants::pi * p.z),
                       -std::sin(2. * ponos::Constants::pi * p.x) * SQR(std::sin(ponos::Constants::pi * p.y))
                           * std::sin(2. * ponos::Constants::pi * p.z),
                       -std::sin(2. * ponos::Constants::pi * p.x) * std::sin(2. * ponos::Constants::pi * p.y)
                           * SQR(std::sin(ponos::Constants::pi * p.z))});
}

} // numerical namespace

