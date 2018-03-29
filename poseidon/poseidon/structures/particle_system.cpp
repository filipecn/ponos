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

#include "poseidon/structures/particle_system.h"
#include <type_traits>

namespace poseidon {

ParticleSystem::ParticleSystem(ponos::PointSetInterface *pointInterface) :
    pointSet_(pointInterface) {
}

size_t ParticleSystem::size() const {
  return pointSet_->size();
}

ponos::Point3 ParticleSystem::position(uint i) const {
  return (*pointSet_)[i];
}

void ParticleSystem::setPosition(uint i, ponos::Point3 p) {
  pointSet_->setPosition(i, p);
}

uint ParticleSystem::add(ponos::Point3 p) {
  return pointSet_->add(p);
}

void ParticleSystem::remove(uint i) {
  pointSet_->remove(i);
}

void ParticleSystem::iterateParticles(const std::function<void(ParticleAccessor&)> &f) {
  pointSet_->iteratePoints([&](uint id, ponos::Point3 p) {
    UNUSED_VARIABLE(p);
    ParticleAccessor pa(*this, id);
    f(pa);
  });
}

void ParticleSystem::search(const ponos::BBox &b, const std::function<void(ParticleAccessor)> &f) {
  pointSet_->search(b, [&](uint id) {
    ParticleAccessor pa(*this, id);
    f(pa);
  });
}

ParticleSystem::~ParticleSystem() = default;

} // poseidon namespace
