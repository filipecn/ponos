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

#ifndef POSEIDON_PARTICLE_SYSTEM_H
#define POSEIDON_PARTICLE_SYSTEM_H

#include <ponos/ponos.h>
#include <vector>

namespace poseidon {

/// Represents a particle system with custom properties.
class ParticleSystem {
public:
  friend struct ParticleAccessor;
  /// Auxiliary struct to easy property access
  struct ParticleAccessor {
    /// \param ps Particle system reference
    /// \param i particle's index
    ParticleAccessor(ParticleSystem &ps, uint i) : id_(i), ps_(ps) {}
    /// \tparam T property type (double, int, uchar)
    /// \param p property index
    /// \return reference to property p
    template<typename T> T &property(uint p) {
      return ps_.property<T>(id_, p);
    }
    /// \return particle's position
    ponos::Point3 position() const {
      return ps_.position(id_);
    }
    /// \return particle's id
    uint id() const {
      return id_;
    }
  private:
    uint id_;
    ParticleSystem &ps_;
  };
  /// \param pointInterface point search structure
  explicit ParticleSystem(
      ponos::PointSetInterface *pointInterface = new ponos::ZPointSet(16));
  ~ParticleSystem();
  /// \return number of active particles
  size_t size() const;
  /// \tparam T property type (double, int, uchar)
  /// \param p property id
  /// \return minimum value of property p
  template<typename T> T minValue(size_t p) const;
  /// \tparam T property type (double, int, uchar)
  /// \param p property id
  /// \return maximum value of property p
  template<typename T> T maxValue(size_t p) const;
  /// \tparam T property type (double, int, uchar)
  /// \param v initial value
  /// \return property id
  template<typename T> uint addProperty(T v);
  /// \tparam T property type (double, int, uchar)
  /// \param i particle index
  /// \param p property index
  /// \return reference to property p of particle i
  template<typename T> T &property(uint i, uint p);
  /// \param i particle's id
  /// \return  particle's position
  ponos::Point3 position(uint i) const;
  /// sets particle position
  /// \param i particle's index
  /// \param p new position
  void setPosition(uint i, ponos::Point3 p);
  // adds a new particle
  /// \param p particle's position
  /// \return particle's index
  uint add(ponos::Point3 p);
  // removes particle **i**
  /// \param i particle's index
  void remove(uint i);
  // SEARCH QUERIES
  /// \param f callback for each particle iterated
  void iterateParticles(const std::function<void(ParticleAccessor &)> &f);
  /// \param b search region
  /// \param f callback for each particle iterated
  void search(const ponos::BBox &b, const std::function<void(ParticleAccessor)> &f);
  /// \tparam T property type
  /// \param p sample point
  /// \param i property id
  /// \param r radius of search region
  /// \param interpolator interpolator
  /// \return interpolated value of property i on sample point p
  template <typename T>
  T gatherProperty(
      ponos::Point3 p, uint i, double r,
      ponos::InterpolatorInterface<ponos::Point3, T> *interpolator);
  /// \tparam T property type
  /// \param p sample point
  /// \param ids list of property ids
  /// \param r radius of search region
  /// \param interpolator interpolator
  /// \return interpolated value of properties on sample point p respective to ids order
  template <typename T>
  std::vector<T>
  gatherProperties(ponos::Point3 p, const std::vector<size_t>& ids, double r,
                 ponos::InterpolatorInterface<ponos::Point3, T> *interpolator);

private:
  std::vector<std::vector<double>> propertiesD_;  ///< [property][particle]
  std::shared_ptr<ponos::PointSetInterface> pointSet_; ///< search structure
  // dummy fields for empty references
  double dummyD_;
};

#include "poseidon/structures/particle_system.inl"

} // poseidon namespace

#endif // POSEIDON_PARTICLE_SYSTEM_H
