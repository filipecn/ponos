#ifndef HELIOS_CORE_FILM_H
#define HELIOS_CORE_FILM_H

namespace helios {

	class Film {
  	public:
	 		Film() {}
			virtual ~Film() {}
			float xResolution, yResolution;
	};

} // helios namespace

#endif // HELIOS_CORE_FILM_H

