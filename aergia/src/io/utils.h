#ifndef AERGIA_SCENE_IO_UTILS_H
#define AERGIA_SCENE_IO_UTILS_H

#include <string>

namespace aergia {

	class RawMesh;

	/* load OBJ
	 * @filename **[in]** file path
	 * @mesh **[out]** output
	 * Loads an OBJ file and stores its contents in **mesh**
	 */
	void loadOBJ(const std::string& filename, RawMesh *mesh);

} // aergia namespace

#endif // AERGIA_SCENE_IO_UTILS_H

