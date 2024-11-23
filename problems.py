import cvxpy as cp
import numpy as np

from tqdm import tqdm

'''
This module contains classes for constructing various types of k-means problems.

Currently implemented is:
	- Uncapacitated k-means (min squared distance to closest center)
	- Hard capacitated k-means (with or without compactness constraints)

They all take the following inputs:
	- The number of districts,
	- The number of points to assign to districts ("gridpoints")
	- Populations of each point
	- The distance between each district center and point.

and may take additional configuration options. They return the problem while setting
other relevant parameters (like the assignment variables) of their object.

'''

# Auxiliary: compactness constraints for hard capacitated k-means. There are two
# types: a relaxed constraint that doesn't need binary variables, and an exact
# constraint that does.

# To make the signatures match up, we use binary variables for both even though
# the relaxed version doesn't need them. The MIP preprocessor should detect this
# and do the right thing. TODO: add these to the object itself and use self.assign
# and/or self.assign_binary as appropriate.

# Both are very slow and introduce a ton of constraints. They scale
# as num districts^2 * num points^2 - it's going to be real fun when
# we get to California...

def HCKM_relaxed_compactness(num_districts, num_gridpoints,
		district_point_dist, assign_binary, show_progress=True):

	compactness_constraints = []
	estimated_num_constraints = num_districts * (num_districts-1) / 2 * \
		num_gridpoints * (num_gridpoints-1)

	if show_progress:
		pbar = tqdm(total = estimated_num_constraints)
		write = lambda string: pbar.write(string)
		update = lambda num_processed: pbar.update(num_processed)
	else:
		write = lambda string: None
		update = lambda num_processed: None

	for district_one in range(num_districts):
		write(str(district_one))
		for district_two in range(district_one+1, num_districts):
			write(f"\t{district_two}")
			for x in range(num_gridpoints):
				x_to_d1 = district_point_dist[district_one][x]
				x_to_d2 = district_point_dist[district_two][x]

				# TODO: Explanation here about how this is a relaxation

				if x_to_d1 > x_to_d2:
					update(num_gridpoints)
					continue

				for y in range(num_gridpoints):
					update(1)
					if x == y: continue
					y_to_d1 = district_point_dist[district_one][y]
					y_to_d2 = district_point_dist[district_two][y]

					if y_to_d1 < y_to_d2: continue

					compactness_constraints.append(
						x_to_d1 * assign_binary[district_one][x] + \
						y_to_d2 * assign_binary[district_two][y] <= \
						x_to_d2 * assign_binary[district_one][x] + \
						y_to_d1 * assign_binary[district_two][y])

					# Another attempt can be found below; it doesn't work,
					# but when I mistyped it as x, y, x, y, which is
					# essentially a NOP, that made cvxpy crash... Maybe
					# make a bug example for this later.
					'''
					compactness_constraints.append(
						assign[district_one][x] + \
						assign[district_two][y] >= \
						assign[district_one][y] + \
						assign[district_two][x])
					'''
	if show_progress:
		pbar.close()

	return compactness_constraints

def HCKM_exact_compactness(num_districts, num_gridpoints,
		district_point_dist, assign_binary, show_progress=True):

	compactness_constraints = []
	estimated_num_constraints = num_districts * (num_districts-1) / 2 * \
		num_gridpoints **2

	if show_progress:
		pbar = tqdm(total = estimated_num_constraints)
		write = lambda string: pbar.write(string)
		update = lambda num_processed: pbar.update(num_processed)
	else:
		write = lambda string: None
		update = lambda num_processed: None

	### Compactness constraints
	### These are from https://www.tandfonline.com/doi/pdf/10.3846/1648-4142.2009.24.274-282

	for district_one in range(num_districts):
		write(str(district_one))
		for district_two in range(district_one+1, num_districts):
			for x in range(num_gridpoints):
				x_to_d1 = district_point_dist[district_one][x]
				x_to_d2 = district_point_dist[district_two][x]

				for y in range(num_gridpoints):
					update(1)
					if x == y: continue
					y_to_d1 = district_point_dist[district_one][y]
					y_to_d2 = district_point_dist[district_two][y]

					# TODO: Strictly speaking, we should be maximally
					# generous since we're working on a low res version
					# of the problem. So we should choose x and y points
					# to minimize (x_to_d1 + y_to_d2 - x_to_d2 - y_to_d1).

					# If we could benefit by swapping d1 and d2's
					# allocations of these points...
					if x_to_d1 + y_to_d2 > x_to_d2 + y_to_d1:
						# then that allocation mustn't happen.
						compactness_constraints.append(
							assign_binary[district_one][x] + \
							assign_binary[district_two][y] <= 1)

	if show_progress:
		pbar.close()

	return compactness_constraints

class HardCapacitatedKMeans:
	def __init__(self):
		self._name = "Hard capacitated k-means"
		self.assign = None
		self.objective_scaling_divisor = 1
		self.get_compactness_constraints = lambda a, b, c, d: []

	@property
	def name(self):
		return self._name

	def create_problem(self, num_districts, num_gridpoints,
		block_populations, district_point_dist):

		'''

		minimize sum over points p in the region of interest:
		    sum over districts i = 1..n:
		(1)        assign[i, p] * dist^2(centerx_i, centery_i, p)

		subject to
		(2)    for all i, p: 0 <= assign[i, p] <= 1
		(2b)	(if using binary variables)
				assign_binary[i, p] >= assign[i, p]
				assign_binary[i, p] <= M * assign[i, p]
		(3)    for all i: (sum over x, y: assign[i, x, y] * pop[x, y]) <= tpop/n
		(4)    for all x, y: (sum over i: assign[i, x, y]) = 1

		If we are also using compactness constraints, we need binary variables
		that are 0 when the corresponding assign variable is zero, 1 when positive.
		We'll define them anyway; if we're not using compactness constraints,
		the solver should just discard the binary variables.

		TODO? A "using compactness constraint" parameter so we don't have
		to do that?

		'''

		sq_district_point_dist = district_point_dist ** 2

		self.assign = cp.Variable((num_districts, num_gridpoints))

		# For compactness constraints.
		self.assign_binary = cp.Variable((num_districts, num_gridpoints),
			boolean=True)

		total_population = int(sum(block_populations))

		# Objective function (LP part 1)
		squared_distances_to_center = 0

		# Unlike uncapacitated, the objective function should not
		# multiply by the block population. (I think?)
		for district_idx in range(num_districts):
			squared_distances_to_center += sq_district_point_dist[district_idx] @ self.assign[district_idx]

		# LP part (2)
		constraints = []
		constraints.append(self.assign <= 1)
		constraints.append(0 <= self.assign)

		# MIP part (2b)
		# TODO: Find a more principled way of determining the big M constant
		# so that any in-practice nonzero assign value is allowed.
		constraints.append(self.assign_binary >= self.assign)
		constraints.append(self.assign_binary <= self.assign * 10000) # HACK!

		# LP part (3)
		# We move the n term to the left-hand side to limit floating point problems.

		pop_constraints = []

		for district_idx in range(num_districts):
			pop_constraint = int(num_districts) * (self.assign[district_idx] @ block_populations) <= int(total_population)
			pop_constraints.append(pop_constraint)

		# LP part (4)

		# (4)    for all gridpoints p: (sum over i: assign[i, p]) = 1
		assign_constraints = []

		for region_pt_idx in range(num_gridpoints):
			assign_constraint = cp.sum(self.assign[:,region_pt_idx]) == 1
			assign_constraints.append(assign_constraint)

		constraints += pop_constraints + assign_constraints + \
			self.get_compactness_constraints(num_districts, num_gridpoints,
				district_point_dist, self.assign_binary)

		prob = cp.Problem(cp.Minimize(squared_distances_to_center), constraints)

		return prob

class UncapacitatedKMeans:
	def __init__(self):
		self._name = "Uncapacitated k-means"
		self.assign = None
		self.objective_scaling_divisor = 1

	@property
	def name(self):
		return self._name

	def create_problem(self, num_districts, num_gridpoints,
		block_populations, district_point_dist):

		'''
		minimize sum over points p in the region of interest:
		    sum over districts i = 1..n:
		(1)        assign[i, x, y] * pop[x, y] * dist^2(centerx_i, centery_i, x, y)

		subject to
		(2)    for all i, x, y: 0 <= assign[i, x, y] <= 1
		(3)    for all x, y: (sum over i: assign[i, x, y]) = 1
		'''

		sq_district_point_dist = district_point_dist ** 2

		self.assign = cp.Variable((num_districts, num_gridpoints))

		# Add a small value so that the objective function never multiplies squared distance
		# by a population of zero.
		adj_block_populations = block_populations * 2**8 + 1

		adj_total_pop = int(sum(adj_block_populations))

		# Objective function (LP part 1)
		squared_distances_to_center = 0

		for district_idx in range(num_districts):
			squared_distances_to_center += (adj_block_populations *
				sq_district_point_dist[district_idx]) @ self.assign[district_idx]

		# do some normalization just so I don't have to deal with extremely large numbers.
		self.objective_scaling_divisor = adj_total_pop * np.mean(sq_district_point_dist)

		# LP part (2)
		constraints = []
		constraints.append(self.assign <= 1)
		constraints.append(0 <= self.assign)

		# LP part (4)

		# (4)    for all x, y: (sum over i: assign[i, x, y]) = 1
		assign_constraints = []

		for region_pt_idx in range(num_gridpoints):
			assign_constraint = cp.sum(self.assign[:,region_pt_idx]) == 1
			assign_constraints.append(assign_constraint)

		constraints += assign_constraints

		prob = cp.Problem(cp.Minimize(squared_distances_to_center), constraints)

		return prob