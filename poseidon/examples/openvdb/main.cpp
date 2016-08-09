#include <openvdb/openvdb.h>
#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointConversion.h>
#include <openvdb_points/tools/PointCount.h>

#include <iostream>

using namespace openvdb::tools;

int main()
{
    // Initialize the OpenVDB and OpenVDB Points library.  This must be called at least
    // once per program and may safely be called multiple times.
    openvdb::initialize();
    openvdb::points::initialize();

    // Create some point positions
    std::vector<openvdb::Vec3f> positions;

    positions.push_back(openvdb::Vec3f(1.2, 4.1, 0.5));
    positions.push_back(openvdb::Vec3f(0.9, 8.1, 3.2));
    positions.push_back(openvdb::Vec3f(-3.6, 1.3, 1.5));
    positions.push_back(openvdb::Vec3f(-3.8, 1.4, 1.51));
    positions.push_back(openvdb::Vec3f(-6.8, -9.1, -3.7));
    positions.push_back(openvdb::Vec3f(1.4, 40302.5, 9.5));

    // Create a linear transform with voxel size of 10.0
    const float voxelSize = 10.0f;
    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(voxelSize);

    // Create the PointDataGrid, position attribute is mandatory
    PointDataGrid::Ptr pointDataGrid = createPointDataGrid<PointDataGrid>(
                    positions, TypedAttributeArray<openvdb::Vec3f>::attributeType(), *transform);

    // Output leaf nodes
    std::cout << "Leaf Nodes: " << pointDataGrid->tree().leafCount() << std::endl;

    // Output point count
    std::cout << "Point Count: " << pointCount(pointDataGrid->tree()) << std::endl;

    // Create a VDB file object.
    openvdb::io::File file("mygrids.vdb");

    // Add the grid pointer to a container.
    openvdb::GridPtrVec grids;
    grids.push_back(pointDataGrid);

    // Write out the contents of the container.
    file.write(grids);
    file.close();
}
