#ifndef HELIOS_CORE_FILTER_H
#define HELIOS_CORE_FILTER_H

namespace helios {

	/* interface
	 * Interface for implementation of various types of filter functions.
	 *
	 * **xWidth** and **yWidth** define points in the axis where the function is zero beyond. These points go to each direction, the overall extent (its _support_) in each direction is **twice** those values.
	 */
	class Filter {
		public:
			/* Constructor
			 * @xw **[in]** X width
			 * @yw **[in]** Y width
			 */
			Filter(float xw, float yw)
				: xWidth(xw), yWidth(yw), invXWidth(1.f / xw), invYWidth(1.f / yw) {}
			virtual ~Filter() {}
			/* compute
			 * @x **[in]** sample position relative to the center of the filter in X direction
			 * @y **[in]** sample position relative to the center of the filter in Y direction
			 *
			 * @return the value of the weight of the sample
			 */
			virtual float evaluate(float x, float y) const = 0;
			const float xWidth, yWidth;
			const float invXWidth, invYWidth;
	};

	/* filter implementation
	 * Equally weights all samples within a square region of the image.
	 */
	class BoxFilter : public Filter {
		/* Constructor
		 * @xw **[in]** X width
		 * @yw **[in]** Y width
		 */
		BoxFilter(float xw, float yw)
			: Filter(xw, yw) {}
		/* @inherit */
		float evaluate(float x, float y) const;
	};
} // helios namespace

#endif // HELIOS_CORE_FILTER_H

