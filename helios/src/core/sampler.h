#ifndef HELIOS_CORE_SAMPLER_H
#define HELIOS_CORE_SAMPLER_H

namespace helios {

	class CameraSample {
  	public:
	 		CameraSample() {}
			virtual ~CameraSample() {}

			float imageX, imageY, time;
	};

} // helios namespace

#endif // HELIOS_CORE_SAMPLER_H

