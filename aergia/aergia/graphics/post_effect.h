// Created by filipecn on 3/2/18.
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
#ifndef AERGIA_POST_EFFECT_H
#define AERGIA_POST_EFFECT_H

#include <aergia/graphics/shader.h>
#include <aergia/io/render_texture.h>

namespace aergia {

/// Applies post rendering effects to final frame
class PostEffect {
public:
  /// \param s [optional | default = no effect] effect
  explicit PostEffect(Shader *s = nullptr);
  /// Apply effect to texture
  /// \param in procedural texture containing input image
  /// \param out result of effect applied
  virtual void apply(const RenderTexture &in, RenderTexture &out);

protected:
  std::shared_ptr<Shader> shader;
};

/// Fast Approximate Anti-Aliasing (FXAA) is an anti-aliasing algorithm created by Timothy Lottes under NVIDIA.
class FXAA : public PostEffect {
public:
  FXAA();
  void apply(const RenderTexture &in, RenderTexture &out);
};

} // aergia namespace

#endif //AERGIA_POST_EFFECT_H
