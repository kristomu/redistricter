from scipy.optimize import minimize_scalar, minimize, basinhopping
import numpy as np

from parse_dbf import get_state_names, get_census_block_data
from spheregeom import *

from region import Region

# Simple implementation of local search weighted k-means on a census block
# level. This takes a district list and weights their areas so that the
# populations are close to equal. I might add a postprocessing step to make
# them even closer to equal later.

# 1. Get coordinates and populations
#	  Use Colorado for now.
colorado = Region("tl_2023_08_tabblock20.dbf")

# Given the pop_square_dists that contains the population-weighted distances
# between district centers and census block centers, and proposed (additive)
# district weights, returns the weighted squared distance matrix.
def weight_populations(pop_square_dists, region_block_populations,
	proposed_weights):

	num_districts = len(proposed_weights)
	proposed_weights_col = proposed_weights.reshape(num_districts, 1)

	# The +1e-9 acts to break ties in order of distance for
	# census blocks that nobody lives in.
	weighted_square_dists = pop_square_dists + proposed_weights_col * \
		(region_block_populations + 1e-9)

	return weighted_square_dists

def fit_kmeans_weights(district_indices, region):
	num_districts = len(district_indices)

	block_populations = region.block_populations
	total_population = region.total_population

	district_block_distances = region.get_district_block_distances(
		district_indices)

	pop_square_dists = district_block_distances**2 * block_populations + \
		district_block_distances**2 * 1e-9 # Break ties by distance.

	alpha = 0.0004
	weights = np.array([1] * num_districts)
	default_weights = True
	record_dev = np.inf
	record_weights = []

	# This function makes it easy to use scipy optimization methods,
	# as an auxiliary step.

	def check_population_dist(proposed_weights):
		weighted_square_dists = weight_populations(pop_square_dists,
			block_populations, proposed_weights)
		associated_districts = np.argmin(weighted_square_dists, axis=0)

		district_pops = np.array([np.sum(block_populations[
			associated_districts == x]) for x in range(num_districts)])

		return np.std(district_pops), district_pops

	# Returns the gradient as well, for global optimization purposes
	def check_population_dist_grad(weights):
		error, district_pops = check_population_dist(weights)

		return error, district_pops - region.total_population/num_districts

	def print_new_minimum(x, f, accepted):
		if accepted:
			print(f"at minimum {f:.4f}, accepted")
		else:
			print(f"at minimum {f:.4f}, rejected")

	for i in range(10):
		stalled = False
		print(f"Iter: {i}/10, record: {record_dev}")
		while not stalled:
			record_dev_before = record_dev

			for j in range(400):
				weighted_square_dists = weight_populations(pop_square_dists,
					block_populations, weights)
				associated_districts = np.argmin(weighted_square_dists, axis=0)

				district_pops = np.array([np.sum(region.block_populations[
					associated_districts == x]) for x in range(num_districts)])

				# Roughly as in Janáček and Gábrišová, though I don't
				# clamp to zero but just rescale so the minimum is zero,
				# since adding a constant to all terms has no effect.
				old_weights = np.copy(weights)

				# If this is directly after the unweighted k-means, use
				# a weight vector proportional to the excess population.
				if default_weights:
					old_weights = np.zeros(num_districts)
					default_weights = False

				new_weights = old_weights + alpha * \
					(district_pops - region.total_population/num_districts)
				new_weights -= np.min(new_weights)

				error, excess = check_population_dist(new_weights)

				if error < record_dev:
					record_weights = new_weights
					record_dev = error
				weights = new_weights

			stalled = record_dev == record_dev_before
			if stalled:
				print(f"\tinternal loop stall: record: {record_dev}")
			else:
				print(f"\tinternal loop: reduced to {record_dev}")

		alpha *= 0.95
		weights = record_weights

		# Do a global optimization step for particularly difficult points.
		global_opt = basinhopping(check_population_dist_grad, x0=weights,
			minimizer_kwargs={"method": "L-BFGS-B", "jac": True}, niter_success=20)
		if global_opt.fun < record_dev:
			weights = global_opt.x
			record_dev = global_opt.fun

	# Recalculate assignment from record.
	weighted_square_dists = weight_populations(pop_square_dists,
			block_populations, record_weights)
	record_association = np.argmin(weighted_square_dists, axis=0)

	district_pops = np.array([np.sum(block_populations[
		record_association == x]) for x in range(num_districts)])

	# Get the sum of squared distances from the centers as the objective.
	district_penalties = [np.sum(pop_square_dists[i][record_association == i])
		for i in range(num_districts)]
	distance_penalty = np.sum(district_penalties) / total_population
	pop_penalty = np.std(district_pops) # standard deviation
	pop_maxmin = np.max(district_pops)-np.min(district_pops)

	return distance_penalty, pop_penalty, pop_maxmin, record_association

def fit_and_print(district_indices, region=colorado):
	distance_penalty, pop_penalty, pop_maxmin, assignment = fit_kmeans_weights(
		district_indices, region)

	region.write_image(f"kmeans_out_{district_indices[0]}_pop{pop_maxmin}.png", assignment, 1000**2)

	print(f"Distance: {distance_penalty:.4f} Pop. std.dev: {pop_penalty:.4f}, max-min: {pop_maxmin} for {str(district_indices)}")

def test():
	# Test some very good (and very bad) district center assignments.

	centers_of_interest = [
		# Good population distributions
		[3117, 126352],
		[24944, 57408, 71309],
		[19997, 23780, 88292, 91381, 117998, 123480, 124476, 130672],

		# Good distances
		[15242, 23733, 63447, 86133, 89922, 104716, 125226, 131351],
		[19042, 23317, 47997, 62936, 69916, 84834, 107600, 125567],
		[17751, 35038, 73583, 74619, 84595, 96107, 107911, 128978],

		# Bad population distributions
		[12415, 19038, 49962, 63421, 89508, 116924, 124294, 130428],
		[1029, 27443, 38794, 57516, 76354, 80712, 98249, 127682],

		# Bad distances
		[90019, 111932, 122082, 125802, 128521, 131978, 133947, 135321],
		[7744, 13507, 20911, 75881, 115307, 116693, 125962, 132804],
		[23255, 23766, 30428, 33463, 41185, 48967, 88287, 131743],
		[11167, 20347, 26270, 29544, 44035, 47757, 49177, 49708],
		[3541, 20614, 30484, 32422, 32522, 32728, 45953, 46664],
		[4826, 13270, 17271, 20414, 23771, 27710, 28701, 39536],
		[3690, 7847, 18287, 23701, 25693, 65679, 88895, 98680]
	]

	# Reference results.
	reference_penalties = {
			(3117, 126352): (12517.8735, 4.0),
			(24944, 57408, 71309): (18303.3651, 14.70),
			(19997, 23780, 88292, 91381, 117998, 123480, 124476, 130672):
				(3936.3581, 20.4),
			(15242, 23733, 63447, 86133, 89922, 104716, 125226, 131351):
				(3490, 69),
			(19042, 23317, 47997, 62936, 69916, 84834, 107600, 125567):
				(3664, 379),
			(17751, 35038, 73583, 74619, 84595, 96107, 107911, 128978):
				(3734, 38),
			(12415, 19038, 49962, 63421, 89508, 116924, 124294, 130428):
				(3202, 16012),
			(1029, 27443, 38794, 57516, 76354, 80712, 98249, 127682):
				(2721, 183596),
			(90019, 111932, 122082, 125802, 128521, 131978, 133947, 135321):
				(36087, 26),
			(7744, 13507, 20911, 75881, 115307, 116693, 125962, 132804):
				(34680, 70),
			(23255, 23766, 30428, 33463, 41185, 48967, 88287, 131743):
				(35981, 206),
			(11167, 20347, 26270, 29544, 44035, 47757, 49177, 49708):
				(19134, 97),
			(3541, 20614, 30484, 32422, 32522, 32728, 45953, 46664):
				(33510, 42),
			(4826, 13270, 17271, 20414, 23771, 27710, 28701, 39536):
				(27554, 45),
			(3690, 7847, 18287, 23701, 25693, 65679, 88895, 98680):
				(34315, 77)}

	for proposed_centers in centers_of_interest:
		distance_penalty, pop_penalty, pop_maxmin, assignment = fit_kmeans_weights(
			proposed_centers, colorado)

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