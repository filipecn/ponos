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
///\file p_mesh.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-02-26
///
///\brief Tools to store/manipulate/access meshes composed of N-sided polygons

#ifndef PONOS_PONOS_PONOS_STRUCTURES_P_MESH_H
#define PONOS_PONOS_PONOS_STRUCTURES_P_MESH_H

#include <ponos/geometry/point.h>

namespace ponos {

struct pmesh_pair_hash {
  template<class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

/// The PMesh structure is based on a half-edge structure that represents
/// a oriented surface. A half-edge represents one side of an edge,
/// containing a particular direction. Each half-edge has a pair that goes
/// on the opposite direction. Since this mesh stores polygons with fixed
/// number of faces, we can save some memory and computation, so each
/// half-edge only stores
///     - the index of the source vertex
///     - the index of the face
///     - the index of its half-edge pair
/// Example: the quad mesh (N = 4) with 2 cells and vertices stored in
///          CCW order, with 7 edges (14 half-edges) and 6 vertices:
///             10           11
///       v0 --------- v3 --------- v5                       2         6
///       |      2     |      7     |
///      9| 3        1 | 4         6| 12     with faces  3        1        5
///       |      0     |      5     |
///       v1 --------- v2 --------- v4                       0         4
///              8           13
///
///     Half-edge data is then stored as:
///     he index  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13
///     ---------------------------------------------------------------------
///     vertex    | 1 | 2 | 3 | 0 | 3 | 2 | 4 | 5 | 2 | 1 | 10 |  3 |  5 |  4
///     face      | 0 | 1 | 2 | 3 | 1 | 4 | 5 | 6 | 0 | 3 |  2 |  6 |  5 |  4
///     pair      | 8 | 4 |10 | 9 | 1 |13 |12 |11 | 0 | 3 |  2 |  7 |  6 |  5
///
/// Since half-edges are sorted and grouped in cells, putting boundary
/// half-edges on the end of the buffer, the index of the cell associated
/// with a half-edge with index he is computed as
///                            he / N
/// while the first half-edge of a cell c is c * N.
/// The edges that form a cell can be easily iterated in order, since the
/// next half-edge of a half-edge he can be computed as
///                  (he - first_he + 1) % N + first_he
/// where first_he = (he / N) * N
/// \tparam N
template<u64 N = 3>
class PMesh {
public:
  // ***********************************************************************
  //                          STRUCTURES
  // ***********************************************************************
  struct HalfData {
    u64 vertex{0}; //!< associated vertex index
    u64 face{0}; //!< associated face index
    u64 pair{0}; //!< opposite half-edge index
  };
  // ***********************************************************************
  //                       CONST ITERATORS
  // ***********************************************************************
  class ConstVertexStar;
  class ConstCellAccessor;
  class ConstVertexAccessor;
  class ConstFaceAccessor;
  class const_vertex_star_iterator {
    friend class PMesh<N>::ConstVertexStar;
  public:
    struct VertexStarElement {
      explicit VertexStarElement(u64 he, const PMesh<N> &mesh) : he_(he), mesh_(mesh) {}
      [[nodiscard]] bool isBoundary() const {
        return he_ / N >= mesh_.cell_count_ || mesh_.half_edges_[he_].pair / N >= mesh_.cell_count_;
      }
      [[nodiscard]] u64 cellIndex() const { return he_ / N; }
      [[nodiscard]] u64 faceIndex() const {
        return mesh_.half_edges_[he_].face;
      }
    private:
      u64 he_;
      const PMesh<N> &mesh_;
    };
    const_vertex_star_iterator &operator++() {
      if (current_he_ == starting_he_) {
        loops_++;
        return *this;
      }
      current_he_ = mesh_.half_edges_[mesh_.previousHalfEdge(current_he_)].pair;
      return *this;
    }
    VertexStarElement operator*() const {
      return VertexStarElement(current_he_, mesh_);
    }
    bool operator==(const const_vertex_star_iterator &other) const {
      return starting_he_ == other.starting_he_ && loops_ == other.loops_;
    }
    bool operator!=(const const_vertex_star_iterator &other) const {
      return loops_ != other.loops_ || starting_he_ != other.starting_he_;
    }
  private:
    const_vertex_star_iterator(const PMesh<N> &mesh, u64 start_he, u64 end_he, u8 loops) :
        loops_{loops}, current_he_(start_he), starting_he_(end_he), mesh_(mesh) {
    }
    u8 loops_{0};
    u64 current_he_{0};
    u64 starting_he_{0};
    const PMesh<N> &mesh_;
  };
  class const_cell_iterator {
    friend class PMesh<N>::ConstCellAccessor;
  public:
    struct CellIteratorElement {
      explicit CellIteratorElement(u64 cell_id, const PMesh<N> &mesh) : index(cell_id), mesh_(mesh) {}
      [[nodiscard]] ponos::point3 centroid() const {
        return mesh_.cellCentroid(index);
      }
      const u64 index;
    private:
      const PMesh<N> &mesh_;
    };
    const_cell_iterator &operator++() {
      if (current_cell_id_ < mesh_.cell_count_)
        current_cell_id_++;
      return *this;
    }
    CellIteratorElement operator*() const {
      return CellIteratorElement(current_cell_id_, mesh_);
    }
    bool operator==(const const_cell_iterator &other) const {
      return current_cell_id_ == other.current_cell_id_;
    }
    bool operator!=(const const_cell_iterator &other) const {
      return current_cell_id_ != other.current_cell_id_;
    }
  private:
    explicit const_cell_iterator(const PMesh<N> &mesh, u64 start) : current_cell_id_{start},
                                                                    mesh_(mesh) {}
    u64 current_cell_id_{0};
    const PMesh<N> &mesh_;
  };
  class const_vertex_iterator {
    friend class PMesh<N>::ConstVertexAccessor;
  public:
    struct VertexIteratorElement {
      explicit VertexIteratorElement(u64 vertex_id, const PMesh<N> &mesh)
          : index(vertex_id), mesh_(mesh) {}
      [[nodiscard]] ponos::point3 position() const {
        return mesh_.vertex_positions_[index];
      }
      [[nodiscard]] ConstVertexStar star() {
        return ConstVertexStar(mesh_, index);
      }
      const u64 index;
    private:
      const PMesh<N> &mesh_;
    };
    const_vertex_iterator &operator++() {
      if (current_vertex_id_ < mesh_.vertexCount())
        current_vertex_id_++;
      return *this;
    }
    VertexIteratorElement operator*() const {
      return VertexIteratorElement(current_vertex_id_, mesh_);
    }
    bool operator==(const const_vertex_iterator &other) const {
      return current_vertex_id_ == other.current_vertex_id_;
    }
    bool operator!=(const const_vertex_iterator &other) const {
      return current_vertex_id_ != other.current_vertex_id_;
    }
  private:
    explicit const_vertex_iterator(const PMesh &mesh, u64 start) : current_vertex_id_{start},
                                                                   mesh_(mesh) {}
    u64 current_vertex_id_{0};
    const PMesh<N> &mesh_;
  };
  class const_face_iterator {
    friend class PMesh<N>::ConstFaceAccessor;
  public:
    struct FaceIteratorElement {
      explicit FaceIteratorElement(u64 face_id, PMesh<N> &mesh)
          : index(face_id), mesh_(mesh) {}
      [[nodiscard]] ponos::point3 centroid() const {
        return mesh_.faceCentroid(index);
      }
      void vertices(u64 &a, u64 &b) const {
        mesh_.faceVertices(index, a, b);
      }
      [[nodiscard]] f32 area() const {
        u64 a, b;
        mesh_.faceVertices(index, a, b);
        return ponos::distance(mesh_.vertex_positions_[a], mesh_.vertex_positions_[b]);
      }
      [[nodiscard]] bool isBoundary() const {
        u64 he = mesh_.interleaved_face_data_.template valueAt<u64>(0, index);
        return he / N >= mesh_.cell_count_ ||
            mesh_.half_edges_[he].pair / N >= mesh_.cell_count_;
      }
      void cells(u64 &a, u64 &b) {
        u64 he = mesh_.interleaved_face_data_.template valueAt<u64>(0, index);
        a = he / N;
        b = mesh_.half_edges_[he].pair / N;
      }
      [[nodiscard]] ponos::vec3 tangent() const {
        u64 a, b;
        mesh_.faceVertices(index, a, b);
        return ponos::normalize(mesh_.vertex_positions_[b] - mesh_.vertex_positions_[a]);
      }
      [[nodiscard]] ponos::vec3 normal() const {
        u64 a, b;
        mesh_.faceVertices(index, a, b);
        auto n = mesh_.vertex_positions_[b] - mesh_.vertex_positions_[a];
        auto tmp = n.x;
        n.x = n.y;
        n.y = -tmp;
        return ponos::normalize(n);
      }
      const u64 index;
    private:
      PMesh<N> &mesh_;
    };
    const_face_iterator &operator++() {
      if (current_face_id_ < mesh_.faceCount())
        current_face_id_++;
      return *this;
    }
    FaceIteratorElement operator*() const {
      return FaceIteratorElement(current_face_id_, mesh_);
    }
    bool operator==(const const_face_iterator &other) const {
      return current_face_id_ == other.current_face_id_;
    }
    bool operator!=(const const_face_iterator &other) const {
      return current_face_id_ != other.current_face_id_;
    }
  private:
    explicit const_face_iterator(const PMesh<N> &mesh, u64 start) : current_face_id_{start},
                                                                    mesh_(mesh) {}
    u64 current_face_id_{0};
    const PMesh<N> &mesh_;
  };
  // ***********************************************************************
  //                           ITERATORS
  // ***********************************************************************
  class VertexStar;
  class CellAccessor;
  class VertexAccessor;
  class FaceAccessor;
  class vertex_star_iterator {
    friend class PMesh<N>::VertexStar;
  public:
    struct VertexStarElement {
      explicit VertexStarElement(u64 he, PMesh<N> &mesh) : he_(he), mesh_(mesh) {}
      [[nodiscard]] bool isBoundary() const {
        return he_ / N >= mesh_.cell_count_ || mesh_.half_edges_[he_].pair / N >= mesh_.cell_count_;
      }
      [[nodiscard]] u64 cellIndex() const { return he_ / N; }
      [[nodiscard]] u64 faceIndex() const {
        return mesh_.half_edges_[he_].face;
      }
    private:
      u64 he_;
      PMesh<N> &mesh_;
    };
    vertex_star_iterator &operator++() {
      if (current_he_ == starting_he_) {
        loops_++;
        return *this;
      }
      current_he_ = mesh_.half_edges_[mesh_.previousHalfEdge(current_he_)].pair;
      return *this;
    }
    VertexStarElement operator*() const {
      return VertexStarElement(current_he_, mesh_);
    }
    bool operator==(const vertex_star_iterator &other) const {
      return starting_he_ == other.starting_he_ && loops_ == other.loops_;
    }
    bool operator!=(const vertex_star_iterator &other) const {
      return loops_ != other.loops_ || starting_he_ != other.starting_he_;
    }
  private:
    vertex_star_iterator(PMesh<N> &mesh, u64 start_he, u64 end_he, u8 loops) :
        loops_{loops}, current_he_(start_he), starting_he_(end_he), mesh_(mesh) {
    }
    u8 loops_{0};
    u64 current_he_{0};
    u64 starting_he_{0};
    PMesh<N> &mesh_;
  };
  class cell_iterator {
    friend class PMesh<N>::CellAccessor;
  public:
    struct CellIteratorElement {
      explicit CellIteratorElement(u64 cell_id, PMesh<N> &mesh) : index(cell_id), mesh_(mesh) {}
      [[nodiscard]] ponos::point3 centroid() const {
        return mesh_.cellCentroid(index);
      }
      const u64 index;
    private:
      PMesh<N> &mesh_;
    };
    cell_iterator &operator++() {
      if (current_cell_id_ < mesh_.cell_count_)
        current_cell_id_++;
      return *this;
    }
    CellIteratorElement operator*() const {
      return CellIteratorElement(current_cell_id_, mesh_);
    }
    bool operator==(const cell_iterator &other) const {
      return current_cell_id_ == other.current_cell_id_;
    }
    bool operator!=(const cell_iterator &other) const {
      return current_cell_id_ != other.current_cell_id_;
    }
  private:
    explicit cell_iterator(PMesh<N> &mesh, u64 start) : current_cell_id_{start},
                                                        mesh_(mesh) {}
    u64 current_cell_id_{0};
    PMesh<N> &mesh_;
  };
  class vertex_iterator {
    friend class PMesh<N>::VertexAccessor;
  public:
    struct VertexIteratorElement {
      explicit VertexIteratorElement(u64 vertex_id, PMesh<N> &mesh)
          : index(vertex_id), mesh_(mesh) {}
      [[nodiscard]] ponos::point3 position() const {
        return mesh_.vertex_positions_[index];
      }
      [[nodiscard]] VertexStar star() {
        return VertexStar(mesh_, index);
      }
      const u64 index;
    private:
      PMesh<N> &mesh_;
    };
    vertex_iterator &operator++() {
      if (current_vertex_id_ < mesh_.vertexCount())
        current_vertex_id_++;
      return *this;
    }
    VertexIteratorElement operator*() const {
      return VertexIteratorElement(current_vertex_id_, mesh_);
    }
    bool operator==(const vertex_iterator &other) const {
      return current_vertex_id_ == other.current_vertex_id_;
    }
    bool operator!=(const vertex_iterator &other) const {
      return current_vertex_id_ != other.current_vertex_id_;
    }
  private:
    explicit vertex_iterator(PMesh &mesh, u64 start) : current_vertex_id_{start},
                                                       mesh_(mesh) {}
    u64 current_vertex_id_{0};
    PMesh<N> &mesh_;
  };
  class face_iterator {
    friend class PMesh<N>::FaceAccessor;
  public:
    struct FaceIteratorElement {
      explicit FaceIteratorElement(u64 face_id, PMesh<N> &mesh)
          : index(face_id), mesh_(mesh) {}
      [[nodiscard]] ponos::point3 centroid() const {
        return mesh_.faceCentroid(index);
      }
      void vertices(u64 &a, u64 &b) const {
        mesh_.faceVertices(index, a, b);
      }
      [[nodiscard]] f32 area() const {
        u64 a, b;
        mesh_.faceVertices(index, a, b);
        return ponos::distance(mesh_.vertex_positions_[a], mesh_.vertex_positions_[b]);
      }
      [[nodiscard]] bool isBoundary() const {
        u64 he = mesh_.interleaved_face_data_.template valueAt<u64>(0, index);
        return he / N >= mesh_.cell_count_ ||
            mesh_.half_edges_[he].pair / N >= mesh_.cell_count_;
      }
      void cells(u64 &a, u64 &b) {
        u64 he = mesh_.interleaved_face_data_.template valueAt<u64>(0, index);
        a = he / N;
        b = mesh_.half_edges_[he].pair / N;
      }
      [[nodiscard]] ponos::vec3 tangent() const {
        u64 a, b;
        mesh_.faceVertices(index, a, b);
        return ponos::normalize(mesh_.vertex_positions_[b] - mesh_.vertex_positions_[a]);
      }
      [[nodiscard]] ponos::vec3 normal() const {
        u64 a, b;
        mesh_.faceVertices(index, a, b);
        auto n = mesh_.vertex_positions_[b] - mesh_.vertex_positions_[a];
        auto tmp = n.x;
        n.x = n.y;
        n.y = -tmp;
        return ponos::normalize(n);
      }
      const u64 index;
    private:
      PMesh<N> &mesh_;
    };
    face_iterator &operator++() {
      if (current_face_id_ < mesh_.faceCount())
        current_face_id_++;
      return *this;
    }
    FaceIteratorElement operator*() const {
      return FaceIteratorElement(current_face_id_, mesh_);
    }
    bool operator==(const face_iterator &other) const {
      return current_face_id_ == other.current_face_id_;
    }
    bool operator!=(const face_iterator &other) const {
      return current_face_id_ != other.current_face_id_;
    }
  private:
    explicit face_iterator(PMesh<N> &mesh, u64 start) : current_face_id_{start},
                                                        mesh_(mesh) {}
    u64 current_face_id_{0};
    PMesh<N> &mesh_;
  };
  // ***********************************************************************
  //                         CONST ACCESSORS
  // ***********************************************************************
  ///
  class VertexStar {
    friend class PMesh<N>;
  public:
    vertex_star_iterator begin() {
      return vertex_star_iterator(mesh_, start_, end_, 0);
    }
    vertex_star_iterator end() {
      return vertex_star_iterator(mesh_, end_, end_, 1);
    }
  private:
    VertexStar(PMesh<N> &mesh, u64 vertex_id) : mesh_{mesh} {
      end_ = mesh_.interleaved_vertex_data_.template valueAt<u64>(0, vertex_id);
      start_ = mesh_.half_edges_[mesh_.previousHalfEdge(end_)].pair;
    }
    u64 start_{0};
    u64 end_{0};
    PMesh<N> &mesh_;
  };
  ///
  class CellAccessor {
    friend class PMesh<N>;
  public:
    cell_iterator begin() {
      return cell_iterator(mesh_, 0);
    }
    cell_iterator end() {
      return cell_iterator(mesh_, mesh_.cell_count_);
    }
  private:
    explicit CellAccessor(PMesh<N> &mesh) : mesh_{mesh} {}
    PMesh<N> &mesh_;
  };
  ///
  class VertexAccessor {
    friend class PMesh<N>;
  public:
    vertex_iterator begin() {
      return vertex_iterator(mesh_, 0);
    }
    vertex_iterator end() {
      return vertex_iterator(mesh_, mesh_.vertexCount());
    }
  private:
    explicit VertexAccessor(PMesh<N> &mesh) : mesh_{mesh} {}
    PMesh<N> &mesh_;
  };
  ///
  class FaceAccessor {
    friend class PMesh<N>;
  public:
    face_iterator begin() {
      return face_iterator(mesh_, 0);
    }
    face_iterator end() {
      return face_iterator(mesh_, mesh_.faceCount());
    }
  private:
    explicit FaceAccessor(PMesh<N> &mesh) : mesh_{mesh} {}
    PMesh<N> &mesh_;
  };
  // ***********************************************************************
  //                           ACCESSORS
  // ***********************************************************************
  ///
  class ConstVertexStar {
    friend class PMesh<N>;
  public:
    const_vertex_star_iterator begin() {
      return const_vertex_star_iterator(mesh_, start_, end_, 0);
    }
    const_vertex_star_iterator end() {
      return const_vertex_star_iterator(mesh_, end_, end_, 1);
    }
  private:
    ConstVertexStar(const PMesh<N> &mesh, u64 vertex_id) : mesh_{mesh} {
      end_ = mesh_.interleaved_vertex_data_.template valueAt<u64>(0, vertex_id);
      start_ = mesh_.half_edges_[mesh_.previousHalfEdge(end_)].pair;
    }
    u64 start_{0};
    u64 end_{0};
    const PMesh<N> &mesh_;
  };
  ///
  class ConstCellAccessor {
    friend class PMesh<N>;
  public:
    const_cell_iterator begin() {
      return const_cell_iterator(mesh_, 0);
    }
    const_cell_iterator end() {
      return const_cell_iterator(mesh_, mesh_.cell_count_);
    }
  private:
    explicit ConstCellAccessor(const PMesh<N> &mesh) : mesh_{mesh} {}
    const PMesh<N> &mesh_;
  };
  ///
  class ConstVertexAccessor {
    friend class PMesh<N>;
  public:
    const_vertex_iterator begin() {
      return const_vertex_iterator(mesh_, 0);
    }
    const_vertex_iterator end() {
      return const_vertex_iterator(mesh_, mesh_.vertexCount());
    }
  private:
    explicit ConstVertexAccessor(const PMesh<N> &mesh) : mesh_{mesh} {}
    const PMesh<N> &mesh_;
  };
  ///
  class ConstFaceAccessor {
    friend class PMesh<N>;
  public:
    const_face_iterator begin() {
      return const_face_iterator(mesh_, 0);
    }
    const_face_iterator end() {
      return const_face_iterator(mesh_, mesh_.faceCount());
    }
  private:
    explicit ConstFaceAccessor(PMesh<N> &mesh) : mesh_{ mesh } {}
    const PMesh <N> &mesh_;
  };
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  ///
  PMesh() {
    interleaved_face_data_.pushField<u64>("half_edge");
    interleaved_vertex_data_.pushField<u64>("half_edge");
  }
  ///
  ~PMesh() = default;
  // ***********************************************************************
  //                           METHODS
  // ***********************************************************************
  ///
  /// \param vertex_positions
  /// \param cells
  /// \param faces_per_cell
  /// \return
  bool buildFrom(const std::vector<ponos::point3> &vertex_positions, const std::vector<u64> &cells) {
    vertex_positions_ = vertex_positions;
    const u64 cell_count = cells.size() / N;
    // temporarily store a map to keep track of edges for edge_pair assignment
    std::unordered_map<std::pair<u64, u64>, u64, pmesh_pair_hash> half_edge_map;
    // keep track of new faces
    u64 face_count = 0;
    // iterate over each cell and create all half_edges
    for (u64 cell_id = 0; cell_id < cell_count; ++cell_id) {
      const u64 cell_index = cell_id * N;
      u64 vertices[N];
      for (u64 i = 0; i < N; ++i)
        vertices[i] = cells[cell_index + i];
      // TODO: check vertex order... assuming CCW
      const u64 half_edge_index_base = half_edges_.size();
      // iterate faces
      for (u32 face = 0; face < N; ++face) {
        // each face of a triangle generates a new half edge which can be the
        // pair of another half edge
        HalfData half_edge;
        half_edge.vertex = vertices[face];
        auto key = std::make_pair(
            std::min(vertices[face], vertices[((face + 1) % N)]),
            std::max(vertices[face], vertices[((face + 1) % N)])
        );
        // the connection between half edges in a pair occurs only once and must be
        // at the moment the second half edge is found. This way we guarantee the
        // half edges of the same triangles are packed together in the array
        if (half_edge_map.find(key) != half_edge_map.end()) {
          // connect pair
          half_edge.pair = half_edge_map[key];
          half_edges_[half_edge.pair].pair = half_edge_index_base + face;
          // connect face (built when the pair was first created)
          half_edge.face = half_edges_[half_edge.pair].face;
          // remove key from map
          half_edge_map.erase(key);
        } else {
          // just add pair to the hash map and go forward
          half_edge_map[key] = half_edge_index_base + face;
          // create the full face associated to this half edge
          half_edge.face = face_count++;
        }
        half_edges_.emplace_back(half_edge);
      }
    }
    // Create boundary half-edges
    // start with any boundary he (the ones left in the half edge map)
    if (!half_edge_map.empty()) {
      //                 end            start  pair
      std::unordered_map<u64, std::pair<u64, u64>> outter_boundary_map;
      for (const auto &boundary_he : half_edge_map)
        if (boundary_he.first.first == half_edges_[boundary_he.second].vertex)
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
        half_edge.face = half_edges_[pair].face;
        half_edge.vertex = vertex;
        half_edges_[pair].pair = half_edges_.size();
        half_edges_.emplace_back(half_edge);
        vertex = outter_boundary_map[vertex].first;
      }
    }
    // Now that we have all the half-edges in hands we can setup memory for cells,
    // faces and vertices
    interleaved_cell_data_.resize(cell_count);
    interleaved_face_data_.resize(face_count);
    interleaved_vertex_data_.resize(vertex_positions.size());
    // iterate over half edges and connect them to cells, faces and vertices
    auto face_he = interleaved_face_data_.field<u64>("half_edge");
    auto vertex_he = interleaved_vertex_data_.field<u64>("half_edge");
    for (u64 hei = 0; hei < half_edges_.size(); ++hei) {
      face_he[half_edges_[hei].face] = hei;
      vertex_he[half_edges_[hei].vertex] = hei;
    }
    cell_count_ = cell_count;
    return true;
  }
  // ***********************************************************************
  //                           DATA
  // ***********************************************************************
  [[nodiscard]] const std::vector<ponos::point3> &positions() const {
    return vertex_positions_;
  }
  ponos::AoS &cellInterleavedData() { return interleaved_cell_data_; }
  ponos::AoS &faceInterleavedData() { return interleaved_face_data_; }
  ponos::AoS &vertexInterleavedData() { return interleaved_vertex_data_; }
  // ***********************************************************************
  //                           METRICS
  // ***********************************************************************
  [[nodiscard]] u64 vertexCount() const {
    return vertex_positions_.size();
  }
  [[nodiscard]] u64 faceCount() const {
    return interleaved_face_data_.size();
  }
  [[nodiscard]] u64 cellCount() const {
    return cell_count_;
  }
  // ***********************************************************************
  //                           GEOMETRY
  // ***********************************************************************
  [[nodiscard]] ponos::point3 cellCentroid(u64 cell_id) const {
    ponos::point3 centroid;
    for (u64 f = 0; f < N; ++f) {
      const auto &p = vertex_positions_[half_edges_[cell_id * N + f].vertex];
      centroid.x += p.x;
      centroid.y += p.y;
      centroid.z += p.z;
    }
    centroid /= N;
    return centroid;
  }
  [[nodiscard]] ponos::point3 faceCentroid(u64 face_id) const {
    u64 a, b;
    faceVertices(face_id, a, b);
    return {
        (vertex_positions_[a].x + vertex_positions_[b].x) * 0.5f,
        (vertex_positions_[a].y + vertex_positions_[b].y) * 0.5f,
        (vertex_positions_[a].z + vertex_positions_[b].z) * 0.5f
    };
  }
  [[nodiscard]] ponos::normal3 cellNormal(u64 cell_id) const {
    // TODO: use the only first 3 vertices is right?
    u64 vertices[3] = {0, 0, 0};
    for (u32 i = 0; i < 3; ++i) {
      vertices[i] = half_edges_[cell_id * 3 + i].vertex;
    }
    auto normal = ponos::normalize(ponos::cross(vertex_positions_[vertices[1]] - vertex_positions_[vertices[0]],
                                                vertex_positions_[vertices[2]] - vertex_positions_[vertices[1]]));
    return ponos::normal3(normal.x, normal.y, normal.z);
  }
  // ***********************************************************************
  //                           TOPOLOGY
  // ***********************************************************************
  [[nodiscard]] inline u64 nextHalfEdge(u64 half_edge_id) const {
    if (half_edge_id >= N * cell_count_) {
      const auto boundary_size = half_edges_.size() - cell_count_ * N;
      const auto boundary_offset = cell_count_ * N;
      return boundary_offset + ((half_edge_id - boundary_offset + 1) % boundary_size);
    }
    const auto half_edge_id_base = (half_edge_id / N) * N;
    return half_edge_id_base + ((half_edge_id - half_edge_id_base + 1) % N);
  }
  [[nodiscard]] inline u64 previousHalfEdge(u64 half_edge_id) const {
    if (half_edge_id >= N * cell_count_) {
      if (half_edge_id - 1 < cell_count_ * N)
        return half_edges_.size() - 1;
      return half_edge_id - 1;
    }
    const auto half_edge_id_base = (half_edge_id / N) * N;
    return half_edge_id_base + ((half_edge_id - half_edge_id_base + N - 1) % N);
  }
  [[nodiscard]] inline u64 halfEdgeFace(u64 half_edge_id) const {
    return half_edges_[half_edge_id].face;
  }
  inline void faceVertices(u64 face_id, u64 &a, u64 &b) const {
    auto he = interleaved_face_data_.valueAt<u64>(0, face_id);
    a = half_edges_[he].vertex;
    b = half_edges_[half_edges_[he].pair].vertex;
  }
  // ***********************************************************************
  //                           ACCESS
  // ***********************************************************************
  VertexStar vertexStar(u64 vertex_id) {
    return VertexStar(*this, vertex_id);
  }
  CellAccessor cells() {
    return CellAccessor(*this);
  }
  VertexAccessor vertices() {
    return VertexAccessor(*this);
  }
  FaceAccessor faces() {
    return FaceAccessor(*this);
  }
  ConstVertexStar vertexStar(u64 vertex_id) const {
    return ConstVertexStar(*this, vertex_id);
  }
  ConstCellAccessor cells() const {
    return ConstCellAccessor(*this);
  }
  ConstVertexAccessor vertices() const {
    return ConstVertexAccessor(*this);
  }
  ConstFaceAccessor faces() const {
    return ConstFaceAccessor(*this);
  }
  // ***********************************************************************
  //                              IO
  // ***********************************************************************
  friend std::ostream &operator<<(std::ostream &os, const PMesh<N> &mesh) {
    os << "PMesh(" << mesh.vertexCount() << " vertices, "
       << mesh.faceCount() << " faces, " << mesh.cellCount() << " " << N << "-sided cells)\n";
    os << "**** Vertices ****\n";
    for (u64 v = 0; v < mesh.vertexCount(); ++v)
      os << "\tVertex #" << v << ": " << mesh.vertex_positions_[v] << std::endl;
    os << "**** Half Edges ****\n";
    for (u64 he = 0; he < mesh.cell_count_; ++he)
      for (u64 i = 0; i < N; ++i)
        os << "\tHalfEdge #" << he * N + i << ": pair " << mesh.half_edges_[he * N + i].pair <<
           " vertex " << mesh.half_edges_[he * N + i].vertex << " face " <<
           mesh.half_edges_[he * N + i].face << " next " << mesh.nextHalfEdge(he * N + 1)
           << std::endl;
    os << "**** boundary ****\n";
    for (u64 he = mesh.cell_count_ * N; he < mesh.half_edges_.size(); ++he)
      os << "\tHalfEdge #" << he << ": pair " << mesh.half_edges_[he].pair <<
         " vertex " << mesh.half_edges_[he].vertex << " face " <<
         mesh.half_edges_[he].face << " next " << mesh.nextHalfEdge(he) << std::endl;
    os << "**** Faces ****\n";
    auto face_he = mesh.interleaved_face_data_.template field<u64>("half_edge");
    for (u64 f = 0; f < mesh.faceCount(); ++f) {
      os << "\tFace #" << f << ": v" <<
         mesh.half_edges_[face_he[f]].vertex << " v";
      os << mesh.half_edges_[mesh.nextHalfEdge(face_he[f])].vertex << std::endl;
    }
    os << "**** Cells ****\n";
    for (u64 c = 0; c < mesh.cellCount(); ++c) {
      os << "\tCell #" << c << ": ";
      auto he = c * N;
      for (u32 f = 0; f < N; ++f)
        os << "f" << mesh.half_edges_[he++].face << " ";
      os << std::endl;
    }
    return os;
  }
  friend std::ofstream &operator<<(std::ofstream &o, const PMesh &mesh) {
    o.write(reinterpret_cast<const char *>(&mesh.cell_count_), sizeof(u64));
    u64 size = mesh.vertex_positions_.size();
    o.write(reinterpret_cast<const char *>(&size), sizeof(u64));
    if (size)
      for (u64 i = 0; i < size; ++i) {
        o.write(reinterpret_cast<const char *>(&mesh.vertex_positions_[i].x), sizeof(real_t));
        o.write(reinterpret_cast<const char *>(&mesh.vertex_positions_[i].y), sizeof(real_t));
        o.write(reinterpret_cast<const char *>(&mesh.vertex_positions_[i].z), sizeof(real_t));
      }
    size = mesh.half_edges_.size();
    o.write(reinterpret_cast<const char *>(&size), sizeof(u64));
    if (size)
      for (u64 i = 0; i < size; ++i) {
        o.write(reinterpret_cast<const char *>(&mesh.half_edges_[i].vertex), sizeof(HalfData::vertex));
        o.write(reinterpret_cast<const char *>(&mesh.half_edges_[i].face), sizeof(HalfData::face));
        o.write(reinterpret_cast<const char *>(&mesh.half_edges_[i].pair), sizeof(HalfData::pair));
      }
    o << mesh.interleaved_face_data_;
    o << mesh.interleaved_face_data_;
    o << mesh.interleaved_cell_data_;
    return o;
  }
  friend std::ifstream &operator>>(std::ifstream &i, PMesh<N> &mesh) {
    mesh = PMesh();
    i.read(reinterpret_cast<char *>(&mesh.cell_count_), sizeof(u64));
    u64 size = 0;
    i.read(reinterpret_cast<char *>(&size), sizeof(u64));
    if (size)
      for (u64 k = 0; k < size; ++k) {
        ponos::point3 p;
        i.read(reinterpret_cast<char *>(&p.x), sizeof(real_t));
        i.read(reinterpret_cast<char *>(&p.y), sizeof(real_t));
        i.read(reinterpret_cast<char *>(&p.z), sizeof(real_t));
        mesh.vertex_positions_.emplace_back(p);
      }
    size = 0;
    i.read(reinterpret_cast<char *>(&size), sizeof(u64));
    if (size)
      for (u64 k = 0; k < size; ++k) {
        HalfData h;
        i.read(reinterpret_cast<char *>(&h.vertex), sizeof(HalfData::vertex));
        i.read(reinterpret_cast<char *>(&h.face), sizeof(HalfData::face));
        i.read(reinterpret_cast<char *>(&h.pair), sizeof(HalfData::pair));
        mesh.half_edges_.emplace_back(h);
      }
    i >> mesh.interleaved_face_data_;
    i >> mesh.interleaved_face_data_;
    i >> mesh.interleaved_cell_data_;
    return i;
  }
private:
  u64 cell_count_{0};
  std::vector<ponos::point3> vertex_positions_;
  std::vector<HalfData> half_edges_;
  // data field values
  ponos::AoS interleaved_cell_data_{};
  ponos::AoS interleaved_vertex_data_{};
  ponos::AoS interleaved_face_data_{};
};

}

#endif //PONOS_PONOS_PONOS_STRUCTURES_N_MESH_H
