/// Copyright (c) 2018, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file n_mesh.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-02-27
///
///\brief Tools to store/manipulate/access meshes composed of N-sided polygons

#include <ponos/structures/n_mesh.h>

namespace ponos {

struct pair_hash {
  template<class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

NMesh::NMesh() {
  interleaved_cell_data_.pushField<u64>("face_count");
  interleaved_face_data_.pushField<u64>("half_edge");
  interleaved_vertex_data_.pushField<u64>("half_edge");
}

NMesh::~NMesh() = default;

NMesh::NMesh(NMesh &&other) noexcept {
  boundary_offset_ = other.boundary_offset_;
  vertex_positions_ = std::move(other.vertex_positions_);
  half_edges_ = std::move(other.half_edges_);
  interleaved_cell_data_ = std::move(other.interleaved_cell_data_);
  interleaved_face_data_ = std::move(other.interleaved_face_data_);
  interleaved_vertex_data_ = std::move(other.interleaved_vertex_data_);
}

NMesh::NMesh(NMesh &other) {
  boundary_offset_ = other.boundary_offset_;
  vertex_positions_ = other.vertex_positions_;
  half_edges_ = other.half_edges_;
  interleaved_cell_data_ = other.interleaved_cell_data_;
  interleaved_face_data_ = other.interleaved_face_data_;
  interleaved_vertex_data_ = other.interleaved_vertex_data_;
}

// ***********************************************************************

//                           OPERATORS
// ***********************************************************************
NMesh &NMesh::operator=(const NMesh &other) {
  boundary_offset_ = other.boundary_offset_;
  vertex_positions_ = other.vertex_positions_;
  half_edges_ = other.half_edges_;
  interleaved_cell_data_ = other.interleaved_cell_data_;
  interleaved_face_data_ = other.interleaved_face_data_;
  interleaved_vertex_data_ = other.interleaved_vertex_data_;
  return *this;
}

NMesh &NMesh::operator=(NMesh &&other) {
  boundary_offset_ = other.boundary_offset_;
  vertex_positions_ = std::move(other.vertex_positions_);
  half_edges_ = std::move(other.half_edges_);
  interleaved_cell_data_ = std::move(other.interleaved_cell_data_);
  interleaved_face_data_ = std::move(other.interleaved_face_data_);
  interleaved_vertex_data_ = std::move(other.interleaved_vertex_data_);
  return *this;
}
// ***********************************************************************
//                           METHODS
// ***********************************************************************
NMesh NMesh::buildFrom(const std::vector<ponos::point3> &vertex_positions, const std::vector<u64> &cells,
                       const std::vector<u64> &face_count_per_cell) {
  NMesh mesh;
  mesh.vertex_positions_ = vertex_positions;
  const u64 cell_count = face_count_per_cell.size();
  // temporarily store a map to keep track of edges for edge_pair assignment
  std::unordered_map<std::pair<u64, u64>, u64, pair_hash> half_edge_map;
  // keep track of new faces
  u64 face_count = 0;
  u64 cell_index = 0;
  mesh.boundary_offset_ = 0;
  // iterate over each cell and create all half_edges
  for (u64 cell_id = 0; cell_id < cell_count; ++cell_id) {
    // TODO: check vertex order... assuming CCW
    const u64 half_edge_index_base = mesh.half_edges_.size();
    // iterate faces
    const u64 cell_face_count = face_count_per_cell[cell_id];
    for (u64 face = 0; face < cell_face_count; ++face) {
      // each face of a cell generates a new half edge which can be the
      // pair of another half edge
      HalfData half_edge;
      half_edge.vertex = cells[cell_index + face];
      half_edge.cell = cell_id;
      auto key = std::make_pair(
          std::min(cells[cell_index + face], cells[cell_index + ((face + 1) % cell_face_count)]),
          std::max(cells[cell_index + face], cells[cell_index + ((face + 1) % cell_face_count)])
      );
      // the connection between half edges in a pair occurs only once and must be
      // at the moment the second half edge is found. This way we guarantee the
      // half edges of the same cell are packed together in the array
      if (half_edge_map.find(key) != half_edge_map.end()) {
        // connect pair
        half_edge.pair = half_edge_map[key];
        mesh.half_edges_[half_edge.pair].pair = half_edge_index_base + face;
        // connect face (built when the pair was first created)
        half_edge.face = mesh.half_edges_[half_edge.pair].face;
        // remove key from map
        half_edge_map.erase(key);
      } else {
        // just add pair to the hash map and go forward
        half_edge_map[key] = half_edge_index_base + face;
        // create the full face associated to this half edge
        half_edge.face = face_count++;
      }
      mesh.half_edges_.emplace_back(half_edge);
    }
    // advance to next cell
    cell_index += cell_face_count;
    mesh.boundary_offset_ += cell_face_count;
  }
  // Create boundary half-edges
  // start with any boundary he (the ones left in the half edge map)
  if (!half_edge_map.empty()) {
    std::unordered_map<u64, std::pair<u64, u64>> outter_boundary_map;
    for (const auto &boundary_he : half_edge_map)
      if (boundary_he.first.first == mesh.half_edges_[boundary_he.second].vertex)
        outter_boundary_map[boundary_he.first.second] =
            std::make_pair(boundary_he.first.first, boundary_he.second);
      else
        outter_boundary_map[boundary_he.first.first] =
            std::make_pair(boundary_he.first.second, boundary_he.second);
    auto vertex = outter_boundary_map.begin()->first;
    for (u64 i = 0; i < outter_boundary_map.size(); ++i) {
      HalfData half_edge;
      auto pair = outter_boundary_map[vertex].second;
      half_edge.pair = pair;
      half_edge.face = mesh.half_edges_[pair].face;
      half_edge.vertex = vertex;
      mesh.half_edges_[pair].pair = mesh.half_edges_.size();
      mesh.half_edges_.emplace_back(half_edge);
      vertex = outter_boundary_map[vertex].first;
    }
  }
  // Now that we have all the half-edges in hands we can setup memory for cells,
  // faces and vertices
  mesh.interleaved_cell_data_.resize(cell_count);
  mesh.interleaved_face_data_.resize(face_count);
  mesh.interleaved_vertex_data_.resize(vertex_positions.size());
  // iterate over half edges and connect them to cells, faces and vertices
  auto face_he = mesh.interleaved_face_data_.field<u64>("half_edge");
  auto vertex_he = mesh.interleaved_vertex_data_.field<u64>("half_edge");
  for (u64 hei = 0; hei < mesh.half_edges_.size(); ++hei) {
    face_he[mesh.half_edges_[hei].face] = hei;
    vertex_he[mesh.half_edges_[hei].vertex] = hei;
  }
  auto cell_he = mesh.interleaved_cell_data_.field<u64>("face_count");
  cell_index = 0;
  for (u64 cid = 0; cid < cell_count; ++cid)
    cell_he[cid] = cell_index, cell_index += face_count_per_cell[cid];
  return mesh;
}

// ***********************************************************************
//                           DATA
// ***********************************************************************
const std::vector<ponos::point3> &NMesh::positions() const {
  return vertex_positions_;
}

ponos::AoS &NMesh::cellInterleavedData() { return interleaved_cell_data_; }

ponos::AoS &NMesh::faceInterleavedData() { return interleaved_face_data_; }

ponos::AoS &NMesh::vertexInterleavedData() { return interleaved_vertex_data_; }

// ***********************************************************************
//                           METRICS
// ***********************************************************************
u64 NMesh::vertexCount() const {
  return vertex_positions_.size();
}
u64 NMesh::faceCount() const {
  return interleaved_face_data_.size();
}
u64 NMesh::cellCount() const {
  return interleaved_cell_data_.size();
}
u64 NMesh::cellFaceCount(u64 cell_id) const {
  auto N = (cell_id == interleaved_cell_data_.size() - 1) ?
           boundary_offset_ : interleaved_cell_data_.valueAt<u64>(0, cell_id + 1);
  return N - interleaved_cell_data_.valueAt<u64>(0, cell_id);
}

}
