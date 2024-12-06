import cvxpy as cp
import numpy as np

from tqdm import tqdm

'''
This module contains classes for constructing various types of k-means problems.

Currently implemented is:
	- Uncapacitated k-means (min squared distance to closest center)
	- Hard capacitated k-means (with or without compactness constraints)

They all take the following inputs:
	- The desired number of districts,
	- The number of points to assign to districts ("gridpoints")
	- Populations of each point
	- The distance between each district center and point.

and may take additional configuration options. They return the problem while setting
other relevant parameters (like the assignment variables) of their object.

The desired number of districts may differ from the actual number of districts
provided. In that case, the problem should specify that only a desired number
is to be chosen out of the ones provided, and the solver has to figurer out
what selection is best.

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

# TODO: Put these in classes, too, and let the k-means constructors
# accept them. Add support for relaxed compactness that doesn't use
# binaries. (so the compactness constraint class should have
# self.requires_assign_binaries or something like.)
# Then try compactness, 100 points, 8-of-16 district centers, and check
# what the solving speed is like. The continuous assign version would
# work as upper bounds, branch and bound style.

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

	### Compactness constraints form Janáček and Gábrišová.
	### https://www.tandfonline.com/doi/pdf/10.3846/1648-4142.2009.24.274-282

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

# From Janáček and Gábrišová, page 278.

def HCKM_thm_4_compactness(num_districts, num_gridpoints,
		district_point_dist, assign_binary):

	# Let v_ik(x) be the difference dist[k, x] - dist[i, x].
	# Let v_ik be the minimum such value over every point x
	# assigned to i.

	# Then districts are compact if -v_ik <= v_ki for all pairs
	# of districts i and k.

	# We use a big M method for this:

	#	v_ik <= (1 - assign_binary[i, x]) * 2*M + dist[k, x] - dist[i, x].

	# I'll tweak this M later and perhaps introduce a way to get
	# a more precise bound for it. We need M to be the greatest value of
	# dist[k,x] - dist[i,x] over every x possible. A looser bound that
	# doesn't depend on the locations of districts is two times the
	# maximum distance between two points.

	M = 1e9 # TBD

	constraints = []
	v = cp.Variable((num_districts, num_districts))

	for i in range(num_districts):
		for k in range(num_districts):
			if i == k:
				continue

			for p in range(num_gridpoints):
				constraints.append(v[i][k] <= (1 - assign_binary[i][p]) * 2*M +
					district_point_dist[k][p] - district_point_dist[i][p])

			constraints.append(-v[i][k] <= v[k][i])

	return constraints

# NOTE about HCKM: Using a very large number of candidate districts and
# a small number of desired districts leads SCIP to almost immediately
# find the proper primal bound. It then spends an extremely long time
# narrowing the dual bound to the primal bound. Perhaps some additional
# technically redundant constraints could be implemented here to speed
# up the process?

# Perhaps we could take some inspiration from Ágoston and Nagy,
# "Mixed integer linear programming formulation for K-means clustering
# problem".

class HardCapacitatedKMeans:
	def __init__(self):
		self._name = "Hard capacitated k-means"
		self.assign = None
		self.objective_scaling_divisor = 1
		self.has_compactness_constraints = False
		self.get_compactness_constraints = None

	@property
	def name(self):
		return self._name

	def create_problem(self, desired_num_districts, num_gridpoints,
		block_populations, district_point_dist):

		'''
		The free variables used are:
			assign[i, p]	How much of point p to assign to district i
			active[i]		Whether district i is used at all: binary
		The input data is:
			dist(centerx_i, centery_i, p)
							Distance from district i center to point p
			tpop			The total population
			n				The desired number of districts
			M				A large constant.

		minimize sum over points p in the region of interest:
		    sum over districts i = 1..n:
		(1)        assign[i, p] * dist^2(centerx_i, centery_i, p)

		subject to
		(2)    for all i, p: 0 <= assign[i, p] <= active[i]
		(2b)	(if using binary variables)
				assign_binary[i, p] >= assign[i, p]
				assign_binary[i, p] <= M * assign[i, p]
		(3)    for all i: (sum over p: assign[i, p] * pop[p]) <= tpop/n
		(4)    for all p: (sum over i: assign[i, p]) = 1
		(5)	   sum over i: active[i] = n

		If we are also using compactness constraints, we need binary variables
		that are 0 when the corresponding assign variable is zero, 1 when positive.
		Apparently the solver isn't clever enough to discard binary variables
		that don't relate to anything, so if we're not using compactness
		constraints, we create a dummy constraint that takes no time to solve.
		(Removing the binary variables outright crashes cvxpy for some reason)

		If the number of desired districts is equal to the number of actual
		districts, then we don't need active variables. While the presolver
		should remove them anyway, this can apparently take a lot of time,
		so we just omit them directly in that case instead.

		'''

		sq_district_point_dist = district_point_dist ** 2

		# Get the number of candidate districts to choose from
		num_districts = len(sq_district_point_dist)

		self.assign = cp.Variable((num_districts, num_gridpoints))

		if desired_num_districts == num_districts:
			self.active = [cp.Constant(1) for _ in range(num_districts)]
		else:
			self.active = cp.Variable(num_districts, boolean=True)

		# For compactness constraints.
		self.assign_binary = cp.Variable((num_districts, num_gridpoints),
			boolean=True)

		total_population = int(sum(block_populations))

		# (1): Define the objective function
		squared_distances_to_center = 0

		# Unlike uncapacitated, the objective function should not
		# multiply by the block population. (I think?)
		for district_idx in range(num_districts):
			squared_distances_to_center += sq_district_point_dist[district_idx] @ self.assign[district_idx]

		# (2) Define the assignment values as fractions, and force to zero if
		#	  the district is inactive.

		constraints = []
		constraints.append(0 <= self.assign)
		for i in range(num_districts):
			constraints.append(self.assign[i] <= self.active[i])

		# (2b) Set binary assignment variables for use with compactness
		#      constraints.

		# TODO: Find a more principled way of determining the big M constant
		# so that any in-practice nonzero assign value is allowed.
		if self.has_compactness_constraints:
			constraints.append(self.assign_binary >= self.assign)
			constraints.append(self.assign_binary <= self.assign * 10000) # HACK!
		else:
			# BUG: For some bizarre reason, not having any assign_binary
			# constraints makes cvxpy crash. The same happens if we remove
			# assign_binary altogether. Something to report?
			constraints.append(self.assign_binary == 1)

		# (3) Enforce the capacity limit on districts.
		# We move the n term to the left-hand side to limit floating point problems.

		pop_constraints = []

		for district_idx in range(num_districts):
			pop_constraint = int(desired_num_districts) * (self.assign[district_idx] @ block_populations) <= int(total_population)
			pop_constraints.append(pop_constraint)

		# (4) Set that every point must be assigned to *someone*.

		# (4)    for all gridpoints p: (sum over i: assign[i, p]) = 1
		assign_constraints = []

		for region_pt_idx in range(num_gridpoints):
			assign_constraint = cp.sum(self.assign[:,region_pt_idx]) == 1
			assign_constraints.append(assign_constraint)

		constraints += pop_constraints + assign_constraints

		if self.has_compactness_constraints:
			constraints += self.get_compactness_constraints(num_districts,
				num_gridpoints, district_point_dist, self.assign_binary)

		# (5): Enforce that the desired number of districts is chosen.

		constraints.append(cp.sum(self.active) == desired_num_districts)

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

	def create_problem(self, desired_num_districts, num_gridpoints,
		block_populations, district_point_dist):

		'''

		The free variables used are:
			assign[i, p]	How much of point p to assign to district i
			active[i]		Whether district i is used at all: binary
		The input data is:
			dist(centerx_i, centery_i, p)
							Distance from district i center to point p
			pop(p)			The number of people closest to point p of the
							given points
			n				The desired number of districts

		minimize sum over points p in the region of interest:
		    sum over districts i = 1..n:
		(1)        assign[i, p] * pop(p) * dist^2(centerx_i, centery_i, p)

		subject to
		(2)    for all i, p: 0 <= assign[i, p] <= active[i]
		(3)    for all p: (sum over i: assign[i, p]) = 1
		(4)	   sum over i: active[i] = n
		'''

		sq_district_point_dist = district_point_dist ** 2

		# Get the number of candidate districts to choose from
		num_districts = len(sq_district_point_dist)

		self.assign = cp.Variable((num_districts, num_gridpoints))
		self.active = cp.Variable(num_districts, boolean=True)

		# Add a small value so that the objective function never multiplies squared distance
		# by a population of zero.
		adj_block_populations = block_populations * 2**8 + 1

		adj_total_pop = int(sum(adj_block_populations))

		# (1): Define the objective function
		squared_distances_to_center = 0

		for district_idx in range(num_districts):
			squared_distances_to_center += (adj_block_populations *
				sq_district_point_dist[district_idx]) @ self.assign[district_idx]

		# do some normalization just so I don't have to deal with extremely large numbers.
		self.objective_scaling_divisor = adj_total_pop * np.mean(sq_district_point_dist)

		# (2) Define the assignment values as fractions, and force to zero if
		#	  the district is inactive.

		constraints = []
		for i in range(num_districts):
			constraints.append(self.assign[i] <= self.active[i])
		constraints.append(0 <= self.assign)

		# (3) Set that every point must be assigned to *someone*.

		# (3)    for all x, y: (sum over i: assign[i, x, y]) = 1
		assign_constraints = []

		for region_pt_idx in range(num_gridpoints):
			assign_constraint = cp.sum(self.assign[:,region_pt_idx]) == 1
			assign_constraints.append(assign_constraint)

		# (4): Enforce that the desired number of districts is chosen.

		constraints.append(cp.sum(self.active) == desired_num_districts)

		constraints += assign_constraints

		prob = cp.Problem(cp.Minimize(squared_distances_to_center), constraints)

		return prob
