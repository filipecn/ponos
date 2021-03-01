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
///\date 2021-02-26
///
///\brief Tools to store/manipulate/access meshes composed of N-sided polygons

#ifndef PONOS_PONOS_PONOS_STRUCTURES_N_MESH_H
#define PONOS_PONOS_PONOS_STRUCTURES_N_MESH_H

#include <ponos/geometry/point.h>
#include <ponos/storage/array_of_structures.h>
#include <ponos/structures/p_mesh.h>

namespace ponos {

/// The NMesh structure is based on a half-edge structure that represents
/// a oriented surface. A half-edge represents one side of an edge,
/// containing a particular direction. Each half-edge has a pair that goes
/// on the opposite direction. Each half-edge stores
///     - the index of the source vertex
///     - the index of its half-edge pair
///     - the index of the associated face
///     - the index of the associated cell
/// Example: The mesh containing a quad (cell 0) and a triangle (cell 1)
///             8             9
///     v0 ------------- v3 -------- v4                 3           5
///      |      3        |     6   /
///     7|0            2 | 4     /      with faces 0          2
///      |      1        |    5/ 10                                   4
///     v1 ------------- v2  /                          1
///             11
///     Each cell stores the index of its first half-edge:
///         cell id | 0 | 1 |
///         -----------------
///         he index| 0 | 4 |
///     Note that the number of sides of a cell i is cell[i+1] - cell[i].
///     Half-edge data is then stored as:
///     he index  | 0 |  1 | 2 | 3 | 4 |  5 | 6 | 7 | 8 | 9 | 10 | 11 |
///     -------------------------------------------------------------
///     vertex    | 0 |  1 | 2 | 3 | 3 |  2 | 4 | 1 | 0 | 3 |  4 |  2 |
///     pair      | 7 | 11 | 4 | 8 | 2 | 10 | 9 | 0 | 3 | 6 |  5 |  1 |
///     face      | 0 |  1 | 2 | 3 | 2 |  4 | 5 | 0 | 3 | 5 |  4 |  1 |
///     cell      | 0 |  0 | 0 | 0 | 1 |  1 | 1 | x | x | x |  x |  x |
/// The edges that form a cell can be easily iterated in order, since the
/// next half-edge of a half-edge he can be computed as
///                  (he - first_he + 1) % N + first_he
/// where first_he = faces_per_cell[half_edges[he].cell] and
///              N = faces_per_cell[he+1] - faces_per_cell[he]
class NMesh {
public:
  // ***********************************************************************
  //                          STRUCTURES
  // ***********************************************************************
  struct HalfData {
    u64 vertex{0}; //!< associated vertex index
    u64 face{0}; //!< associated face index
    u64 pair{0}; //!< opposite half-edge index
    u64 cell{Constants::greatest<u64>()}; //!< associated cell index
  };
  // ***********************************************************************
  //                         Const Iterators
  // ***********************************************************************
  class ConstVertexStar;
  class const_vertex_star_iterator {
    friend class NMesh::ConstVertexStar;
  public:
    struct VertexStarElement {
      explicit VertexStarElement(u64 he, const NMesh &mesh) : he_(he), mesh_(mesh) {}
      [[nodiscard]] bool isBoundary() const {
        return he_ >= mesh_.boundary_offset_;
      }
      [[nodiscard]] u64 cellIndex() const { return mesh_.half_edges_[he_].cell; }
      [[nodiscard]] u64 faceIndex() const {
        return mesh_.half_edges_[he_].face;
      }
    private:
      u64 he_;
      const NMesh &mesh_;
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
    const_vertex_star_iterator(const NMesh &mesh, u64 start_he, u64 end_he, u8 loops) :
        loops_{loops}, current_he_(start_he), starting_he_(end_he), mesh_(mesh) {
    }
    u8 loops_{0};
    u64 current_he_{0};
    u64 starting_he_{0};
    const NMesh &mesh_;
  };
  class ConstCellAccessor;
  class const_cell_iterator {
    friend class NMesh::ConstCellAccessor;
  public:
    struct CellIteratorElement {
      explicit CellIteratorElement(u64 cell_id, const NMesh &mesh) : index(cell_id), mesh_(mesh) {}
      [[nodiscard]] ponos::point3 centroid() const {
        return mesh_.cellCentroid(index);
      }
      const u64 index;
    private:
      const NMesh &mesh_;
    };
    const_cell_iterator &operator++() {
      if (current_cell_id_ < mesh_.interleaved_cell_data_.size())
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
    explicit const_cell_iterator(const NMesh &mesh, u64 start) : current_cell_id_{start},
                                                           mesh_(mesh) {}
    u64 current_cell_id_{0};
    const NMesh &mesh_;
  };
  class ConstVertexAccessor;
  class const_vertex_iterator {
    friend class NMesh::ConstVertexAccessor;
  public:
    struct VertexIteratorElement {
      explicit VertexIteratorElement(u64 vertex_id, const NMesh &mesh)
          : index(vertex_id), mesh_(mesh) {}
      [[nodiscard]] ponos::point3 position() const {
        return mesh_.vertex_positions_[index];
      }
      [[nodiscard]] ConstVertexStar star() {
        return ConstVertexStar(mesh_, index);
      }
      const u64 index;
    private:
      const NMesh &mesh_;
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
    explicit const_vertex_iterator(const NMesh &mesh, u64 start) : current_vertex_id_{start},
                                                             mesh_(mesh) {}
    u64 current_vertex_id_{0};
    const NMesh &mesh_;
  };
  class ConstFaceAccessor;
  class const_face_iterator {
    friend class NMesh::ConstFaceAccessor;
  public:
    struct FaceIteratorElement {
      explicit FaceIteratorElement(u64 face_id, const NMesh &mesh)
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
        return he >= mesh_.boundary_offset_ || mesh_.half_edges_[he].pair >= mesh_.boundary_offset_;
      }
      void cells(u64 &a, u64 &b) {
        u64 he = mesh_.interleaved_face_data_.template valueAt<u64>(0, index);
        a = mesh_.half_edges_[he].cell;
        b = mesh_.half_edges_[mesh_.half_edges_[he].pair].cell;
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
      const NMesh &mesh_;
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
    explicit const_face_iterator(const NMesh &mesh, u64 start) : current_face_id_{start},
                                                           mesh_(mesh) {}
    u64 current_face_id_{0};
    const NMesh &mesh_;
  };
  // ***********************************************************************
  //                           Iterators
  // ***********************************************************************
  class VertexStar;
  class vertex_star_iterator {
    friend class NMesh::VertexStar;
  public:
    struct VertexStarElement {
      explicit VertexStarElement(u64 he, NMesh &mesh) : he_(he), mesh_(mesh) {}
      [[nodiscard]] bool isBoundary() const {
        return he_ >= mesh_.boundary_offset_;
      }
      [[nodiscard]] u64 cellIndex() const { return mesh_.half_edges_[he_].cell; }
      [[nodiscard]] u64 faceIndex() const {
        return mesh_.half_edges_[he_].face;
      }
    private:
      u64 he_;
      NMesh &mesh_;
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
    vertex_star_iterator(NMesh &mesh, u64 start_he, u64 end_he, u8 loops) :
        loops_{loops}, current_he_(start_he), starting_he_(end_he), mesh_(mesh) {
    }
    u8 loops_{0};
    u64 current_he_{0};
    u64 starting_he_{0};
    NMesh &mesh_;
  };
  class CellAccessor;
  class cell_iterator {
    friend class NMesh::CellAccessor;
  public:
    struct CellIteratorElement {
      explicit CellIteratorElement(u64 cell_id, NMesh &mesh) : index(cell_id), mesh_(mesh) {}
      [[nodiscard]] ponos::point3 centroid() const {
        return mesh_.cellCentroid(index);
      }
      const u64 index;
    private:
      NMesh &mesh_;
    };
    cell_iterator &operator++() {
      if (current_cell_id_ < mesh_.interleaved_cell_data_.size())
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
    explicit cell_iterator(NMesh &mesh, u64 start) : current_cell_id_{start},
                                                     mesh_(mesh) {}
    u64 current_cell_id_{0};
    NMesh &mesh_;
  };
  class VertexAccessor;
  class vertex_iterator {
    friend class NMesh::VertexAccessor;
  public:
    struct VertexIteratorElement {
      explicit VertexIteratorElement(u64 vertex_id, NMesh &mesh)
          : index(vertex_id), mesh_(mesh) {}
      [[nodiscard]] ponos::point3 position() const {
        return mesh_.vertex_positions_[index];
      }
      [[nodiscard]] VertexStar star() {
        return VertexStar(mesh_, index);
      }
      const u64 index;
    private:
      NMesh &mesh_;
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
    explicit vertex_iterator(NMesh &mesh, u64 start) : current_vertex_id_{start},
                                                       mesh_(mesh) {}
    u64 current_vertex_id_{0};
    NMesh &mesh_;
  };
  class FaceAccessor;
  class face_iterator {
    friend class NMesh::FaceAccessor;
  public:
    struct FaceIteratorElement {
      explicit FaceIteratorElement(u64 face_id, NMesh &mesh)
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
        return he >= mesh_.boundary_offset_ || mesh_.half_edges_[he].pair >= mesh_.boundary_offset_;
      }
      void cells(u64 &a, u64 &b) {
        u64 he = mesh_.interleaved_face_data_.template valueAt<u64>(0, index);
        a = mesh_.half_edges_[he].cell;
        b = mesh_.half_edges_[mesh_.half_edges_[he].pair].cell;
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
      NMesh &mesh_;
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
    explicit face_iterator(NMesh &mesh, u64 start) : current_face_id_{start},
                                                     mesh_(mesh) {}
    u64 current_face_id_{0};
    NMesh &mesh_;
  };
  // ***********************************************************************
  //                          CONST ACCESSORS
  // ***********************************************************************
  ///
  class ConstVertexStar {
    friend class NMesh;
  public:
    const_vertex_star_iterator begin() {
      return const_vertex_star_iterator(mesh_, start_, end_, 0);
    }
    const_vertex_star_iterator end() {
      return const_vertex_star_iterator(mesh_, end_, end_, 1);
    }
  private:
    ConstVertexStar(const NMesh &mesh, u64 vertex_id) : mesh_{mesh} {
      end_ = mesh_.interleaved_vertex_data_.template valueAt<u64>(0, vertex_id);
      start_ = mesh_.half_edges_[mesh_.previousHalfEdge(end_)].pair;
    }
    u64 start_{0};
    u64 end_{0};
    const NMesh &mesh_;
  };
  ///
  class ConstCellAccessor {
    friend class NMesh;
  public:
    const_cell_iterator begin() {
      return const_cell_iterator(mesh_, 0);
    }
    const_cell_iterator end() {
      return const_cell_iterator(mesh_, mesh_.interleaved_cell_data_.size());
    }
  private:
    explicit ConstCellAccessor(const NMesh &mesh) : mesh_{mesh} {}
    const NMesh &mesh_;
  };
  ///
  class ConstVertexAccessor {
    friend class NMesh;
  public:
    const_vertex_iterator begin() {
      return const_vertex_iterator(mesh_, 0);
    }
    const_vertex_iterator end() {
      return const_vertex_iterator(mesh_, mesh_.vertexCount());
    }
  private:
    explicit ConstVertexAccessor(const NMesh &mesh) : mesh_{mesh} {}
    const NMesh &mesh_;
  };
  ///
  class ConstFaceAccessor {
    friend class NMesh;
  public:
    const_face_iterator begin() {
      return const_face_iterator(mesh_, 0);
    }
    const_face_iterator end() {
      return const_face_iterator(mesh_, mesh_.faceCount());
    }
  private:
    explicit ConstFaceAccessor(const NMesh &mesh) : mesh_{ mesh } {}
    const NMesh &mesh_;
  };
  // ***********************************************************************
  //                           ACCESSORS
  // ***********************************************************************
  ///
  class VertexStar {
    friend class NMesh;
  public:
    vertex_star_iterator begin() {
      return vertex_star_iterator(mesh_, start_, end_, 0);
    }
    vertex_star_iterator end() {
      return vertex_star_iterator(mesh_, end_, end_, 1);
    }
  private:
    VertexStar(NMesh &mesh, u64 vertex_id) : mesh_{mesh} {
      end_ = mesh_.interleaved_vertex_data_.template valueAt<u64>(0, vertex_id);
      start_ = mesh_.half_edges_[mesh_.previousHalfEdge(end_)].pair;
    }
    u64 start_{0};
    u64 end_{0};
    NMesh &mesh_;
  };
  ///
  class CellAccessor {
    friend class NMesh;
  public:
    cell_iterator begin() {
      return cell_iterator(mesh_, 0);
    }
    cell_iterator end() {
      return cell_iterator(mesh_, mesh_.interleaved_cell_data_.size());
    }
  private:
    explicit CellAccessor(NMesh &mesh) : mesh_{mesh} {}
    NMesh &mesh_;
  };
  ///
  class VertexAccessor {
    friend class NMesh;
  public:
    vertex_iterator begin() {
      return vertex_iterator(mesh_, 0);
    }
    vertex_iterator end() {
      return vertex_iterator(mesh_, mesh_.vertexCount());
    }
  private:
    explicit VertexAccessor(NMesh &mesh) : mesh_{mesh} {}
    NMesh &mesh_;
  };
  ///
  class FaceAccessor {
    friend class NMesh;
  public:
    face_iterator begin() {
      return face_iterator(mesh_, 0);
    }
    face_iterator end() {
      return face_iterator(mesh_, mesh_.faceCount());
    }
  private:
    explicit FaceAccessor(NMesh &mesh) : mesh_{mesh} {}
    NMesh &mesh_;
  };
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  ///
  NMesh();
  ///
  ~NMesh();
  NMesh(NMesh &&other) noexcept;
  // ***********************************************************************
  //                          STATIC METHODS
  // ***********************************************************************
  ///
  /// \param vertex_positions
  /// \param cells
  /// \param face_count_per_cell
  /// \return
  static NMesh buildFrom(const std::vector<ponos::point3> &vertex_positions, const std::vector<u64> &cells,
                         const std::vector<u64> &face_count_per_cell = {3});
  ///
  /// \tparam N
  /// \param mesh
  /// \return
  template<u64 N>
  static NMesh buildDualFrom(PMesh<N> &mesh) {
    std::vector<ponos::point3> dual_vertices;
    std::vector<u64> dual_cells;
    std::vector<u64> face_count_per_dual_cell;
    // Cell centroids become vertices
    for (auto cell : mesh.cells())
      dual_vertices.template emplace_back(cell.centroid());
    // Each vertex spawns a new cell that is built from the vertex star
    // For each vertex, setup a new cell
    std::unordered_map<u64, u64> face_centers;
    for (u64 vertex_id = 0; vertex_id < mesh.vertexCount(); ++vertex_id) {
      // iterate vertex star
      u64 face_count = 0;
      for (auto s: mesh.vertexStar(vertex_id)) {
        // if face is boundary we need to create a vertex from it
        if (s.isBoundary()) {
          if (!face_centers.count(s.faceIndex())) {
            face_centers[s.faceIndex()] = dual_vertices.size();
            dual_vertices.emplace_back(mesh.faceCentroid(s.faceIndex()));
          }
          dual_cells.emplace_back(face_centers[s.faceIndex()]);
          face_count++;
        }
        if (s.cellIndex() < mesh.cellCount()) {
          dual_cells.emplace_back(s.cellIndex());
          face_count++;
        }
      }
      // create cell
      face_count_per_dual_cell.emplace_back(face_count);
    }
    return buildFrom(dual_vertices, dual_cells, face_count_per_dual_cell);
  }
  // ***********************************************************************
  //                           DATA
  // ***********************************************************************
  [[nodiscard]] const std::vector<ponos::point3> &positions() const;
  ponos::AoS &cellInterleavedData();
  ponos::AoS &faceInterleavedData();
  ponos::AoS &vertexInterleavedData();
  // ***********************************************************************
  //                           METRICS
  // ***********************************************************************
  [[nodiscard]] u64 vertexCount() const;
  [[nodiscard]] u64 faceCount() const;
  [[nodiscard]] u64 cellCount() const;
  [[nodiscard]] u64 cellFaceCount(u64 cell_id) const;
  // ***********************************************************************
  //                           GEOMETRY
  // ***********************************************************************
  [[nodiscard]] ponos::point3 cellCentroid(u64 cell_id) const {
    ponos::point3 centroid;
    auto N = cellFaceCount(cell_id);
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
    if (half_edge_id >= boundary_offset_) {
      const auto boundary_size = half_edges_.size() - boundary_offset_;
      return boundary_offset_ + ((half_edge_id - boundary_offset_ + 1) % boundary_size);
    }
    const auto cell_id = half_edges_[half_edge_id].cell;
    const auto half_edge_id_base = interleaved_cell_data_.valueAt<u64>(0, cell_id);
    u64 N = (cell_id == interleaved_cell_data_.size() - 1) ?
            boundary_offset_ - half_edge_id_base :
            interleaved_cell_data_.valueAt<u64>(0, cell_id + 1) - half_edge_id_base;
    return half_edge_id_base + ((half_edge_id - half_edge_id_base + 1) % N);
  }
  [[nodiscard]] inline u64 previousHalfEdge(u64 half_edge_id) const {
    if (half_edge_id >= boundary_offset_) {
      if (half_edge_id - 1 < boundary_offset_)
        return half_edges_.size() - 1;
      return half_edge_id - 1;
    }
    const auto cell_id = half_edges_[half_edge_id].cell;
    const auto half_edge_id_base = interleaved_cell_data_.valueAt<u64>(0, cell_id);
    u64 N = (cell_id == interleaved_cell_data_.size() - 1) ?
            boundary_offset_ - half_edge_id_base :
            interleaved_cell_data_.valueAt<u64>(0, cell_id + 1) - half_edge_id_base;
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
  ConstFaceAccessor faces() const{
    return ConstFaceAccessor(*this);
  }
  // ***********************************************************************
  //                              IO
  // ***********************************************************************
  friend std::ostream &operator<<(std::ostream &os, const NMesh &mesh) {
    os << "PMesh(" << mesh.vertexCount() << " vertices, "
       << mesh.faceCount() << " faces, " << mesh.cellCount() << " N-sided cells)\n";
    os << "**** Vertices ****\n";
    for (u64 v = 0; v < mesh.vertexCount(); ++v)
      os << "\tVertex #" << v << ": " << mesh.vertex_positions_[v] << std::endl;
    os << "**** Half Edges (sorted by cell) ****\n";
    for (u64 cid = 0; cid < mesh.interleaved_cell_data_.size(); ++cid) {
      const u64 N = mesh.cellFaceCount(cid);
      const u64 he_base = mesh.interleaved_cell_data_.valueAt<u64>(0, cid);
      for (u64 i = 0; i < N; ++i)
        os << "\tHalfEdge #" << he_base + i << ": pair " << mesh.half_edges_[he_base + i].pair <<
           " vertex " << mesh.half_edges_[he_base + i].vertex << " face " <<
           mesh.half_edges_[he_base + i].face << " next " << mesh.nextHalfEdge(he_base + 1)
           << std::endl;
    }
    os << "**** boundary ****\n";
    for (u64 he = mesh.boundary_offset_; he < mesh.half_edges_.size(); ++he)
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
      auto he = mesh.interleaved_cell_data_.valueAt<u64>(0, c);
      const auto N = mesh.cellFaceCount(c);
      for (u32 f = 0; f < N; ++f)
        os << "f" << mesh.half_edges_[he++].face << " ";
      os << std::endl;
    }
    return os;
  }
private:
  u64 boundary_offset_{0};
  std::vector<ponos::point3> vertex_positions_;
  std::vector<HalfData> half_edges_;
  // data field values
  ponos::AoS interleaved_cell_data_{};
  ponos::AoS interleaved_vertex_data_{};
  ponos::AoS interleaved_face_data_{};
};

}

#endif //PONOS_PONOS_PONOS_STRUCTURES_N_MESH_H
