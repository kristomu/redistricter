from scipy.optimize import minimize_scalar, minimize, basinhopping
import numpy as np

from parse_dbf import get_state_names, get_census_block_data
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

def fit_kmeans_weights(district_indices):
	num_districts = len(district_indices)

	block_populations = np.array(
		[block["population"] for block in block_data])
	total_population = np.sum(block_populations)
	block_latlongs = np.array(
		[[block["lat"], block["long"]] for block in block_data])

	district_latlongs = [block_latlongs[i] for i in district_indices]
	district_block_distances = np.array([haversine_center(dl, block_latlongs)
		for dl in district_latlongs])

	pop_square_dists = district_block_distances**2 * block_populations + \
		district_block_distances**2 * 1e-9 # Break ties by distance.

	alpha = 1
	weights = np.array([1] * num_districts)
	record_dev = np.inf
	record_weights = []

	# This function makes it easy to use scipy optimization methods,
	# but they tend only to be useful for very bad fits. See below.

	def check_population_dist(proposed_weights):
		weighted_square_dists = (pop_square_dists.T * proposed_weights).T
		associated_districts = np.argmin(weighted_square_dists, axis=0)

		district_pops = np.array([np.sum(block_populations[
			associated_districts == x]) for x in range(num_districts)])

		return np.std(district_pops)

	def print_new_minimum(x, f, accepted):
		if accepted:
			print(f"at minimum {f:.4f}, accepted")
		else:
			print(f"at minimum {f:.4f}, rejected")

	for j in range(10):
		print(f"Pass {j}")
		for i in range(150):
			old_weights = np.copy(weights)
			weighted_square_dists = (pop_square_dists.T * weights).T
			associated_districts = np.argmin(weighted_square_dists, axis=0)

			district_pops = np.array([np.sum(block_populations[associated_districts == x])
				for x in range(num_districts)])
			std_dev = np.std(district_pops)

			if std_dev < record_dev:
				record_dev = std_dev
				record_weights = np.copy(weights)

			# Update the weights based on current imbalances.
			# There probably exist better update schedules, but this seems OK
			# for now. Janáček and Gábrišová uses an additive update.
			weights = (1-alpha) * weights + alpha * weights * district_pops/np.sum(district_pops)
			if np.sum(weights) == 0:
				weights = np.array([1] * num_districts)

			weights = weights / np.sum(weights)

		weights = np.copy(record_weights)
		alpha = alpha * 0.9

	# We could do something like this to try to find an optimum using scipy.
	# This greatly improves bad fits (where std. devs are in the order of tens of
	# thousands), but such fits are not very useful anyway, and the search would
	# slow down the process for good fits, so I've left it disabled by default.

	use_global_optimizer = False

	if use_global_optimizer:
		global_opt = basinhopping(check_population_dist, x0=record_weights,
			minimizer_kwargs={"method": "Powell"}, niter_success=4,
			callback=print_new_minimum)
		if global_opt.fun < record_dev:
			record_weights = global_opt.x/np.sum(global_opt.x)
			record_dev = global_opt.fun

	# Recalculate assignment from record.
	weighted_square_dists = (pop_square_dists.T * record_weights).T
	record_association = np.argmin(weighted_square_dists, axis=0)

	district_pops = np.array([np.sum(block_populations[
		record_association == x]) for x in range(num_districts)])

	# Get the sum of squared distances from the centers as the objective.
	district_penalties = [np.sum(pop_square_dists[i][record_association == i])
		for i in range(num_districts)]
	distance_penalty = np.sum(district_penalties) / total_population
	pop_penalty = np.std(district_pops) # standard deviation
	pop_maxmin = np.max(district_pops)-np.min(district_pops)

	return distance_penalty, pop_penalty, pop_maxmin

def fit_and_print(district_indices):
	distance_penalty, pop_penalty = fit_kmeans_weights(district_indices)

	print(f"Distance: {distance_penalty:.4f} Pop. std.dev: {pop_penalty:.4f} for {str(proposed_centers)}")

def test():
	# Test some very good (and very bad) district center assignments.

	centers_of_interest = [
		# Good population distributions
		[3117, 126352],
		[24944, 57408, 71309],
		[19997, 23780, 88292, 91381, 117998, 123480, 124476, 130672],

		# Good distances
		[19042, 23317, 47997, 62936, 69916, 84834, 107600, 125567],
		[1029, 27443, 38794, 57516, 76354, 80712, 98249, 127682],
		[17751, 35038, 73583, 74619, 84595, 96107, 107911, 128978],

		# Bad population distributions
		[11167, 20347, 26270, 29544, 44035, 47757, 49177, 49708],
		[3541, 20614, 30484, 32422, 32522, 32728, 45953, 46664],
		[4826, 13270, 17271, 20414, 23771, 27710, 28701, 39536],
		[7744, 13507, 20911, 75881, 115307, 116693, 125962, 132804],

		# Bad distances
		[90019, 111932, 122082, 125802, 128521, 131978, 133947, 135321],
		[23255, 23766, 30428, 33463, 41185, 48967, 88287, 131743],
		[3690, 7847, 18287, 23701, 25693, 65679, 88895, 98680]
	]

	# Reference results.
	reference_penalties = {
			(3117, 126352): (16849.429, 1.0),
			(24944, 57408, 71309): (29376.828, 6.94),
			(19997, 23780, 88292, 91381, 117998, 123480, 124476, 130672):
				(5080.133, 9.61),
			(19042, 23317, 47997, 62936, 69916, 84834, 107600, 125567):
				(3951.410, 54.78),
			(1029, 27443, 38794, 57516, 76354, 80712, 98249, 127682):
				(4052.803, 49.78),
			(17751, 35038, 73583, 74619, 84595, 96107, 107911, 128978):
				(4079.641, 28.23),
			(11167, 20347, 26270, 29544, 44035, 47757, 49177, 49708):
				(20996.375, 317632.62),
			(3541, 20614, 30484, 32422, 32522, 32728, 45953, 46664):
				(41045.028, 310802.97),
			(4826, 13270, 17271, 20414, 23771, 27710, 28701, 39536):
				(31694.395, 284331.82),
			(7744, 13507, 20911, 75881, 115307, 116693, 125962, 132804):
				(37584.978, 275723.31),
			(90019, 111932, 122082, 125802, 128521, 131978, 133947, 135321):
				(49930.074, 104790.34),
			(23255, 23766, 30428, 33463, 41185, 48967, 88287, 131743):
				(40949.199, 88523.20),
			(3690, 7847, 18287, 23701, 25693, 65679, 88895, 98680):
				(37696.509, 108042.33)}

	for proposed_centers in centers_of_interest:
		distance_penalty, pop_penalty, pop_maxmin = fit_kmeans_weights(proposed_centers)

		print(f"Distance: {distance_penalty:.4f} Pop. std.dev: {pop_penalty:.4f}, max-min: {pop_maxmin} for {str(proposed_centers)}")

		reference_dist, reference_pop = reference_penalties[
				tuple(proposed_centers)]
		
		conditions = [":-)", ":-)"]
		if reference_dist > distance_penalty * 1.1:
			conditions[0] = ":-D"
		if reference_dist < distance_penalty * 0.9:
			conditions[0] = ":-("
		if reference_pop > pop_penalty * 1.1:
			conditions[1] = ":-D"
		if reference_pop < pop_penalty * 0.9:
			conditions[1] = ":-("

		print(f"Test {proposed_centers[0]}...: distance: {conditions[0]}, population: {conditions[1]}")

test()
