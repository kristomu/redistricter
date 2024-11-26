# Given an aspect ratio and a target number of points, find a pair of numbers
# whose product is within 1% of the target, and aspect ratio is as close as
# possible to the target.

# We do this by finding the optimum second number of the tuple given the
# first number; then we do a bracketed root search over the first number.

import numpy as np
from scipy.optimize import brentq

def _find_width(height, target_product, target_aspect_ratio):
	'''
	Since our product deviation target is 1%, the optimum
	must satisfy
		height * x >= target_product * 0.99
		height * x <= target_product * 1.01
	and unless one of these bounds is binding, must satisfy
		(x-1) / height <= aspect_ratio
		(x+1) / height >= aspect_ratio.
	Thus, the optimum, measured in distance from the correct aspect ratio,
	must be one of
		ceil(target_product * 0.99 / height)
		floor(target_product * 1.01 / height)
		floor(target_aspect_ratio * height)
		ceil(target_aspect_ratio * height)
	and we can check them all.

	The returned value is the optimum, as well as the signed difference
	between the aspect ratio based on the quantized values, and the real
	aspect ratio.

	'''

	candidates = np.array([
		np.floor(target_aspect_ratio * height),
		np.ceil(target_aspect_ratio * height),
		np.floor(target_product * 1.01 / height),
		np.ceil(target_product * 0.99 / height)])

	candidates = candidates[candidates >= target_product * 0.99 / height]
	candidates = candidates[candidates <= target_product * 1.01 / height]

	# height is greater than the target product alone, so there exists
	# no integer that can meet the product deviation target.
	if len(candidates) == 0:
		return 0, -np.inf

	empirical_ratios = candidates / height - target_aspect_ratio
	optimum_idx = np.argmin(np.fabs(empirical_ratios))

	return candidates[optimum_idx], empirical_ratios[optimum_idx]

def grid_dimensions(target_num_points, target_aspect_ratio):

	fractional_height = brentq(
		lambda height: _find_width(height, target_num_points, target_aspect_ratio)[1],
		1, target_num_points)

	height = None
	width = None
	record_quality = np.inf

	for candidate_height in (np.floor(fractional_height), np.ceil(fractional_height)):
		candidate_width, quality = _find_width(candidate_height,
			target_num_points, target_aspect_ratio)
		if np.fabs(quality) < record_quality:
			record_quality = quality
			height = candidate_height
			width = candidate_width
	
	return (height, width, record_quality)