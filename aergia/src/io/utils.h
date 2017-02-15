#ifndef AERGIA_SCENE_IO_UTILS_H
#define AERGIA_SCENE_IO_UTILS_H

#include <ponos.h>

#include <string>

namespace aergia {

	/* load OBJ
	 * @filename **[in]** file path
	 * @mesh **[out]** output
	 * Loads an OBJ file and stores its contents in **mesh**
	 */
	void loadOBJ(const std::string& filename, ponos::RawMesh *mesh);

} // aergia namespace

#endif // AERGIA_SCENE_IO_UTILS_H

