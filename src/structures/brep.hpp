#pragma once

#include "geometry/point.h"

#include <vector>

namespace ponos {
	/*
	l_s     r_p
		 \   /
		  \ /
		   x b
			 |
		l	 |  r
			 |
			 x a
			/ \
		 /   \
	l_p     r_s
	*/
	template<class T, int D>
		class Brep {
			public:
				struct Vertex {
					Point<T, D> pos;
					std::vector<int> edges;
					Vertex() {}
					Vertex(Point<T, D> p) {
						pos = p;
					}
				};
				struct Edge {
					// vertices
					int a, b;
					// faces
					int l, r;
					// left traverse
					int leftPred, leftSucc;
					// right traverse
					int rightPred, rightSucc;
					Edge() {
						a = b = l = r = leftPred = leftSucc = rightPred = rightSucc = -1;
					}
					Edge(size_t _a, size_t _b) {
						a = _a;
						b = _b;
						l = r = leftPred = leftSucc = rightPred = rightSucc = -1;
					}
				};

				int addVertex(Vertex v);
				int addVertex(Point<T, D> p);
				void removeVertex(size_t i);
				int addEdge(Edge e);
				int addEdge(size_t a, size_t b);
				void removeEdge(size_t i);
				int addFace(const std::vector<int>& vs);
				void removeFace(size_t i);

				std::vector<Vertex> vertices;
				std::vector<int> faces;
				std::vector<Edge> edges;

			private:
				// update vertex edges information
				void updateVertex(size_t v);
		};

	template<class T, int D>
		int Brep<T, D>::addVertex(Brep<T, D>::Vertex v) {
			vertices.emplace_back(v);
			return static_cast<int>(vertices.size()) - 1;
		}
	template<class T, int D>
		int Brep<T, D>::addVertex(Point<T, D> p) {
			vertices.emplace_back(p);
			return static_cast<int>(vertices.size()) - 1;
		}

	template<class T, int D>
		void Brep<T, D>::removeVertex(size_t i) { }

	template<class T, int D>
		int Brep<T, D>::addEdge(Edge e) {
			edges.emplace_back(e);
			return static_cast<int>(edges.size()) - 1;
		}

	template<class T, int D>
		int Brep<T, D>::addEdge(size_t a, size_t b) {
			edges.emplace_back(a, b);
			updateVertex(a);
			updateVertex(b);
			return static_cast<int>(edges.size()) - 1;
		}

	template<class T, int D>
		void Brep<T, D>::removeEdge(size_t i) { }

	template<class T, int D>
		int Brep<T, D>::addFace(const std::vector<int>& vs) {
			/*int newFace = faces.size();
			bool manifold = true;
			for(int v : vs) {
				if(edges[vertices[v].edge].l >= 0) {
					manifold = false;
					break;
				}
			}
			if(!manifold)
				return -1;
			for(int v : vs)
				edges[vertices[v].edge].l = newFace;
			faces.emplace_back(vs[0]);*/
			return 0;
		}

	template<class T, int D>
		void Brep<T, D>::removeFace(size_t i) { }

	template<class T, int D>
		void Brep<T, D>::updateVertex(size_t v) {

		}
} // ponos namespace
