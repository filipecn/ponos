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

#include <aergia/ui/trackball_interface.h>

namespace aergia {

void TrackballInterface::draw() { modes[curMode]->draw(tb); }

void TrackballInterface::buttonRelease(CameraInterface &camera, int button,
                                       ponos::Point2 p) {
  UNUSED_VARIABLE(button);
  modes[curMode]->stop(tb, camera, p);
}

void TrackballInterface::buttonPress(const CameraInterface &camera, int button,
                                     ponos::Point2 p) {
  switch (button) {
  case GLFW_MOUSE_BUTTON_LEFT:
    curMode = 2;
    break;
  case GLFW_MOUSE_BUTTON_MIDDLE:
    curMode = 1;
    break;
  case GLFW_MOUSE_BUTTON_RIGHT:
    curMode = 0;
    break;
  }
  modes[curMode]->start(tb, camera, p);
}

void TrackballInterface::mouseMove(CameraInterface &camera, ponos::Point2 p) {
  modes[curMode]->update(tb, camera, p);
}

void TrackballInterface::mouseScroll(CameraInterface &camera, ponos::vec2 d) {
  if (curMode != 3) {
    curMode = 3;
  }
  modes[curMode]->update(tb, camera, d);
}

} // aergia namespace
