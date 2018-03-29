// Created by filipecn on 3/3/18.
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

#include <cstddef>
#include <functional>
#include "display_renderer.h"

namespace aergia {

DisplayRenderer::DisplayRenderer(size_t w, size_t h) : curBuffer_(0) {
  attributes_.target = GL_TEXTURE_2D;
  attributes_.type = GL_UNSIGNED_BYTE;
  attributes_.internalFormat = GL_RGBA8;
  attributes_.format = GL_RGBA;
  attributes_.width = w;
  attributes_.height = h;
  buffers_.resize(2);
  needsResize_ = true;
}

void DisplayRenderer::addEffect(PostEffect *e) {
  effects_.emplace_back(e);
}

void DisplayRenderer::process(const std::function<void()> &f) {
  resize(attributes_.width, attributes_.height);
  // render image to the first framebuffer
  buffers_[curBuffer_]->render(f);
  for (auto &effect : effects_) {
    effect->apply(*buffers_[curBuffer_].get(), *buffers_[(curBuffer_ + 1) % 2].get());
    curBuffer_ = (curBuffer_ + 1) % 2;
  }
}

void DisplayRenderer::render() {
  resize(attributes_.width, attributes_.height);
  if(!screenShader_)
    screenShader_.reset(new Shader(ShaderManager::instance().loadFromTexts(
        AERGIA_NO_VAO_VS, nullptr, AERGIA_NO_VAO_FS)));
  // render to display
  buffers_[curBuffer_]->bind(GL_TEXTURE0);
  screenShader_->begin();
  screenShader_->setUniform("tex", 0);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  screenShader_->end();
}

void DisplayRenderer::resize(size_t w, size_t h) {
  if (needsResize_) {
    attributes_.width = w;
    attributes_.height = h;
    for (size_t i = 0; i < 2; i++)
      buffers_[i].reset(new RenderTexture(attributes_, parameters_));
  }
  needsResize_ = false;
}

} // aergia namespace
