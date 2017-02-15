#ifndef PONOS_GEOMETRY_SHAPE_H
#define PONOS_GEOMETRY_SHAPE_H

namespace ponos {

  enum class ShapeType {SHAPE, POLYGON, SPHERE, CAPSULE, MESH};

	class Shape {
  	public:
	 		Shape() {
        type = ShapeType::SHAPE;
      }
			virtual ~Shape() {}
      ShapeType type;
	};

} // ponos namespace

#endif // PONOS_GEOMETRY_SHAPE_H
