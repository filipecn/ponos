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

#ifndef AERGIA_UI_TRACKBALL_INTERFACE_H
#define AERGIA_UI_TRACKBALL_INTERFACE_H

#include <aergia/scene/camera.h>
#include <aergia/ui/track_mode.h>
#include <aergia/ui/trackball.h>

#include <ponos/ponos.h>
#include <vector>

namespace aergia {

class TrackballInterface {
public:
  TrackballInterface() {
    modes.emplace_back(new RotateMode());
    modes.emplace_back(new ZMode());
    modes.emplace_back(new PanMode());
    modes.emplace_back(new ScaleMode());
    curMode = 0;
  }

  virtual ~TrackballInterface() {}

  void draw();

  void buttonRelease(CameraInterface &camera, int button, ponos::Point2 p);
  void buttonPress(const CameraInterface &camera, int button, ponos::Point2 p);
  void mouseMove(CameraInterface &camera, ponos::Point2 p);
  void mouseScroll(CameraInterface &camera, ponos::vec2 d);

  Trackball tb;

protected:
  size_t curMode;
  std::vector<TrackMode *> modes;
};

} // aergia namespace

#endif // AERGIA_UI_TRACKBALL_INTERFACE_H
