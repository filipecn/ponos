// Created by filipecn on 2018-12-12.
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

#ifndef HELIOS_H_TRANSFORM_H
#define HELIOS_H_TRANSFORM_H

#include <helios/core/interaction.h>
#include <helios/geometry/bounds.h>
#include <helios/geometry/h_ray.h>
#include <ponos/geometry/transform.h>

namespace helios {

class HTransform : public ponos::Transform {
public:
  /// Applies transform to ray, handling numerical error
  /// \param r ray
  /// \return transformed ray
  inline HRay operator()(const HRay &r) const;
  HRay operator()(const HRay &r, ponos::vec3f *oError, ponos::vec3f *dError) const;
  /// Applies transform to bounds
  /// \param b bounds
  /// \return transformed bounds
  bounds3f operator()(const bounds3f &b) const;
  /// Applies transform to a point and computes its absolute error
  /// \tparam T data type
  /// \param p point
  /// \param pError **[out]** accumulated error in **p** due to transformation
  /// \return transformed point
  template <typename T>
  ponos::Point3<T> operator()(const ponos::Point3<T> &p,
                              ponos::Vector3<T> *pError) const;
  /// Applies transform to a point carrying error and computes its absolute
  /// error \tparam T data type \param p point \param pError accumulated error
  /// in **p** \param pTError **[out]** accumulated error in **p** after
  /// transform \return transformed point
  template <typename T>
  ponos::Point3<T> operator()(const ponos::Point3<T> &p,
                              const ponos::Vector3<T> &pError,
                              ponos::Vector3<T> *pTError) const;
  /// Applies transform to a vector and computes its absolute error
  /// \tparam T data type
  /// \param v vector
  /// \param vError **[out]** accumulated error in **v** due to transformation
  /// \return transformed vector
  template <typename T>
  ponos::Vector3<T> operator()(const ponos::Vector3<T> &v,
                               ponos::Vector3<T> *vError) const;
  /// Applies transform to a vector carrying error and computes its absolute
  /// error
  /// \tparam T data type
  /// \param v vector
  /// \param vError accumulated error in **v**
  /// \param vTError **[out]** accumulated error in **v** after transform
  /// \return transformed point
  template <typename T>
  ponos::Vector3<T> operator()(const ponos::Vector3<T> &v,
                               const ponos::Vector3<T> &vError,
                               ponos::Vector3<T> *vTError) const;
  /// Applies transform to surface interaction members
  /// \param si surface interaction
  /// \return surface interactions with members transformed
  SurfaceInteraction operator()(const SurfaceInteraction &si) const;
};

} // namespace helios

#endif // HELIOS_H_TRANSFORM_H
