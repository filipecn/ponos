// Created by filipecn on 8/17/18.
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

#include <aergia/io/graphics_display.h>
#include "nanogui_screen.h"

namespace aergia {

NanoGUIScreen::NanoGUIScreen() {
  auto& gd = GraphicsDisplay::instance();
  gd.registerMouseFunc([&](double x, double y) {
    this->cursorPosCallbackEvent(x, y);
  });
  gd.registerButtonFunc([&](int button, int action, int modifiers) {
    this->mouseButtonCallbackEvent(button, action, modifiers);
  });
  gd.registerKeyFunc([&](int key, int scancode, int action, int mods) {
    this->keyCallbackEvent(key, scancode, action, mods);
  });
  gd.registerCharFunc([&](unsigned int codepoint) {
    this->charCallbackEvent(codepoint);
  });
  gd.registerDropFunc([&](int count, const char **filenames) {
    this->dropCallbackEvent(count, filenames);
  });
  gd.registerScrollFunc([&](double x, double y) {
    this->scrollCallbackEvent(x, y);
  });
  gd.registerResizeFunc([&](int width, int height) {
    this->resizeCallbackEvent(width, height);
  });
  gd.registerRenderFunc([&]() {
    this->drawContents();
    this->drawWidgets();
  });
  this->initialize(gd.getGLFWwindow(), true);
}

}
