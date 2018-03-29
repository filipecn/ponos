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

#ifndef AERGIA_DISPLAY_RENDERER_H
#define AERGIA_DISPLAY_RENDERER_H

#include <aergia/io/render_texture.h>
#include <aergia/graphics/post_effect.h>

namespace aergia {

/// Renders to the display. Allows multiple post-effects. Post effects are applied
/// on the same order they were added.
class DisplayRenderer {
public:
  DisplayRenderer(size_t w, size_t h);
  /// \param e post effect.
  void addEffect(PostEffect *e);
  /// \param f render callback of the original frame
  void process(const std::function<void()> &f);
  void render();
  /// \param w width in pixels
  /// \param h height in pixels
  void resize(size_t w, size_t h);

private:
  bool needsResize_;
  size_t curBuffer_;
  std::vector<std::shared_ptr<PostEffect>> effects_;
  TextureAttributes attributes_;
  TextureParameters parameters_;
  std::vector<std::shared_ptr<RenderTexture>> buffers_;
  std::shared_ptr<Shader> screenShader_;
};

} // aergia namespace

#endif //AERGIA_DISPLAY_RENDERER_H