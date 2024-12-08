from parse_dbf import get_state_names, get_census_block_data
import numpy as np

from spheregeom import *

# Simple implementation of local search weighted k-means on a census block
# level. This takes a district list and weights their areas so that the
# populations are close to equal. I might add a postprocessing step to make
# them even closer to equal later.

# 1. Get coordinates and populations
#	  Use Colorado for now.
state_names = get_state_names()
block_data = get_census_block_data("tl_2023_08_tabblock20.dbf",
	state_names)

# 2. Convert to 3D points with fixed altitude; and 3. create a bounding box
mean_radius = 6371 # mean distance from the center of the Earth

# Bounding box coordinates are in latitude/longitude, not radian format
# For later use.
minimum_point = None
maximum_point = None

for block in block_data:
	coords = np.array([block["lat"], block["long"]])

	radian_coords = np.radians(coords)

	block["coords"] = LLHtoECEF_rad(
		radian_coords[0], radian_coords[1], mean_radius)

	if minimum_point is None:
		minimum_point = coords
					    
	if maximum_point is None:
		maximum_point = coords
								    
	minimum_point = np.minimum(minimum_point, coords)
	maximum_point = np.maximum(maximum_point, coords)


def fit_kmeans_weights(district_indices, verbose=False):

	num_districts = len(district_indices)

	block_populations = np.array(
		[block["population"] for block in block_data])
	total_population = np.sum(block_populations)
	block_latlongs = np.array(
		[[block["lat"], block["long"]] for block in block_data])

	district_latlongs = [block_latlongs[i] for i in district_indices]

	district_block_distances = np.array([haversine_center(dl, block_latlongs)
		for dl in district_latlongs])
	pop_square_dists = district_block_distances**2 * block_populations

	alpha = 1
	weights = np.array([1] * num_districts)
	record = np.inf
	record_weights = []
	record_association = []

	for j in range(10):
		print(f"Pass {j}")
		for i in range(100):
			old_weights = np.copy(weights)
			weighted_square_dists = (pop_square_dists.T * weights).T

			associated_districts = np.argmin(weighted_square_dists, axis=0)

			district_pops = np.array([np.sum(block_populations[associated_districts == x])
				for x in range(num_districts)])

			# There probably exists better update schedules. Find one later.
			# Janáček and Gábrišová uses an additive update. I could also use
			# some kind of line search, e.g. golden section, here.
			weights = (1-alpha) * weights + alpha * weights * district_pops/np.sum(district_pops)
			weights = weights / np.sum(weights)

			std_dev = np.std(district_pops)

			if verbose:
				print(f"Iteration {i}: performance: max-min: {np.max(district_pops)-np.min(district_pops)}, "
					f"std dev.: {std_dev}")

			if std_dev < record:
				record = std_dev
				record_weights = np.copy(weights)
				record_association = np.copy(associated_districts)

		weights = np.copy(record_weights)
		alpha = alpha * 0.9

	# Get the sum of squared distances from the centers as the objective.
	district_penalties = [np.sum(pop_square_dists[i][record_association == i])
		for i in range(num_districts)]
	distance_penalty = np.sum(district_penalties) / total_population
	pop_penalty = record # standard deviation

	return distance_penalty, pop_penalty

district_indices = [18937, 19979, 74671, 83626, 97369, 107272, 123089, 130918]
#[1100, 60990, 64540, 65189, 84319, 84798, 88665, 90398]

distance_penalty, pop_penalty = fit_kmeans_weights(district_indices)

print(f"Distance: {distance_penalty:.4f} Pop. std.dev: {pop_penalty:.4f} for {str(district_indices)}")
