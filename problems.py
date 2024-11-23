import cvxpy as cp
import numpy as np

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

class HardCapacitatedKMeans:
	def __init__(self):
		self._name = "Hard capacitated k-means"
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
		(1)        assign[i, p] * dist^2(centerx_i, centery_i, p)

		subject to
		(2)    for all i, p: 0 <= assign[i, p] <= 1
		(3)    for all i: (sum over x, y: assign[i, x, y] * pop[x, y]) <= tpop/n
		(4)    for all x, y: (sum over i: assign[i, x, y]) = 1

		'''

		sq_district_point_dist = district_point_dist ** 2

		self.assign = cp.Variable((num_districts, num_gridpoints))

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

		constraints += pop_constraints + assign_constraints

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