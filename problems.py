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

# Auxiliary: compactness constraints for hard capacitated k-means. These
# enforce contiguity of the districts and may also enforce more limiting
# notions of compactness.

class NoCompactness:
	# Null object, implements nothing
	def get_constraints(self):
		return []
	def has_constraints(self):
		return False
	def report(self):
		return None

# From Janáček and Gábrišová, theorem 4.
# https://www.tandfonline.com/doi/pdf/10.3846/1648-4142.2009.24.274-282

# So named because it enforces the constraint that no pair of districts
# can swap points belonging to each other and mutually benefit in terms
# of distance.

class SwapCompactness:
	def has_constraints(self):
		return True

	def get_constraints(self, num_districts, num_gridpoints,
		district_point_dist, assign_binary):

		# Let v_ik(x) be the difference dist[k, x] - dist[i, x].
		# Let v_ik be the minimum such value over every point x
		# assigned to i.

		# Then districts are compact if -v_ik <= v_ki for all pairs
		# of districts i and k.

		# We use a big M method for this:

		#	v_ik <= (1 - assign_binary[i, x]) * 2*M + dist[k, x] - dist[i, x].

		# I'll tweak this M later and perhaps introduce a way to get
		# a more precise bound for it. It would seem clear that with this
		# choice of M, the constraint can't make a difference if assign[i][p]
		# is 0, since even in the worst case, 2*M - dist[k][p] - dist[i][p] >
		# max p: dist[k][p] - dist[i][p]. But I'm not quite sure about my
		# own reasoning.

		M = np.max(district_point_dist)

		constraints = []
		v = cp.Variable((num_districts, num_districts))

		for i in range(num_districts):
			print(i)
			for k in range(num_districts):
				if i == k:
					continue

				for p in range(num_gridpoints):
					constraints.append(v[i][k] <= (1 - assign_binary[i][p]) * 2*M +
						district_point_dist[k][p] - district_point_dist[i][p])

				constraints.append(-v[i][k] <= v[k][i])

		return constraints

	def report(self):
		return None

# This emulates a "Voronoi" constraint where each point closer to A than
# to B is barred from being assigned to B, but where each district also
# has a weight that determines its "distance offset" on the points. This
# should emulate the result of using an iterative method like the moments
# of inertia method by Hess et al., but finding the actual global optimum
# instead of a local one.

# https://pubsonline.informs.org/doi/abs/10.1287/opre.13.6.998

# Currently rather hacky.

# It could also potentially be ported to center-less if I do that
# later, but I'm not sure. Let's do one thing at a time.

# Forcing all weights to be zero (or equal) would have a nice physical
# interpretation: "each district is assigned the points closer to it
# than to any other district". But it would almost certainly mean
# giving up on hard capacities.

# The constraints work like this:
# If a point p is associated to district i and we have infinite resolution,
# then we would want
# 	(weight[i] + dist[i, p]**2) * pop[p] < (weight[k] + dist[k, p]**2) * pop[p]

# However, as we're (likely) using a lower resolution problem, it's possible
# that the bisector (boundary) between i and k cut through the same grid
# square, which would then make the constraint impossible to satisfy. So
# we probably need a relaxation, and I'll try this one:

# If p and q are two points and
#	d(i, p)^2 * pop[p] < d(i, q)^2 * pop[q],
#	d(k, q)^2 * pop[q] < d(k, p)^2 * pop[p],
# and q is claimed by i, then p cannot be claimed by k, so
#	(weight[i] + dist[i, q]**2) * pop[q] < (weight[k] + dist[k, p]**2) * pop[p]
# or in big M terms:
#	(weight[i] + dist[i, q]**2) * pop[q] <= 
#		(weight[k] + dist[k, p]**2) * pop[p] + (1 - assign_binary[i, q]) * M.

# WLOG we can make p and q neighboring points. To do this efficiently we're
# going to need some kind of Delaunay triangulation or to pass additional
# information through about who the neighbors of a point is. I'm not going
# to do that yet.

# There may be better ways, e.g. using adjacent census blocks allocated
# to the same grid square...

# And still it outright fails on some cases, e.g.
# [1029, 27443, 38794, 57516, 76354, 80712, 98249, 127682]
# with 250 gridpoints. TODO: Find out what's going on. And find out why
# the weights are nonsensical (e.g. try to force weights from kmeans.py
# on a case that *does* work, and see if they work or if we get something
# infeasible...)

# When it does work, it seems to enforce the compactness well enough,
# but the weights aren't right.

from spheregeom import *

class NeighborVoronoiCompactness:
	def __init__(self, grid_latlon, district_latlon, grid_populations):
		# Find (i, j, p, q) combinations that can be used to establish
		# the constraint above.
		# Try to do this without Delaunay triangulations first because
		# the numerical precision is not the best. See if it works
		# without introducing additional bugs.

		num_districts = len(district_latlon)
		num_gridpoints = len(grid_latlon)

		self.admissible_points = []
		self.grid_populations = grid_populations

		for i in tqdm(range(num_districts)):
			for k in range(num_districts):
				if i == k: continue

				for p in range(num_gridpoints):
					i_dist_p = haversine_np(
						district_latlon[i][0], district_latlon[i][1],
						grid_latlon[p][0], grid_latlon[p][1])
					k_dist_p = haversine_np(
						district_latlon[k][0], district_latlon[k][1],
						grid_latlon[p][0], grid_latlon[p][1])
					# Pick the point where i_dist_q and k_dist_p are
					# as close as possible.
					record_distance = np.inf
					record_idx = (-1, -1)

					for q in range(num_gridpoints):
						if p == q: continue

						i_dist_q = haversine_np(
							district_latlon[i][0], district_latlon[i][1],
							grid_latlon[q][0], grid_latlon[q][1])
						k_dist_q = haversine_np(
							district_latlon[k][0], district_latlon[k][1],
							grid_latlon[q][0], grid_latlon[q][1])

						if i_dist_p >= i_dist_q: continue
						if k_dist_q >= k_dist_p: continue
						if np.fabs(i_dist_q - k_dist_p) >= record_distance: continue

						record_distance = np.fabs(i_dist_q - k_dist_p)
						record_idx = (p, q)

					if record_idx == (-1, -1):
						continue

					self.admissible_points.append((i, k, record_idx[0],
						record_idx[1]))

	def has_constraints(self):
		return True

	def get_constraints(self, num_districts, num_gridpoints,
		district_point_dist, assign_binary):

		self.weight = cp.Variable(num_districts)

		constraints = []

		M = np.sum(self.grid_populations) * np.max(district_point_dist)**2

		for i, k, p, q in self.admissible_points:
			constraints.append((self.grid_populations[q] + 1e-9) * (self.weight[i] + district_point_dist[i][q]**2) <= \
				(self.grid_populations[p] + 1e-9) * (self.weight[k] + district_point_dist[k][p]**2) + \
				(1 - assign_binary[i][q]) * M)

		constraints.append(self.weight >= 0)

		return constraints

	def report(self):
		print([w.value for w in self.weight])

# One-coordinate Voronoi compactness, using a "slack factor" g,
# which simulates neighbors, to get
#	weight(A) - weight(X) - g < d(p, center_X)^2 - d(p, center_A)^2

# If only one district claims a point, then the unoffsetted
# weight[A] + d(A, p)**2 < weight[B] + d(B, p)**2 can be used, no problem
# (because it works in the fully disaggregated case and aggregated data
# looks just like disaggregated data). However, the weights may be distorted
# somewhat as a result. So the problem arises when there's a fractional claim.

# Other potential speedups that hold for exclusive assignments but not
# fractional ones include

# -dist(i,k)^2 <= weight(i) - weight(k) <= dist(i, k)^2
#	because due to the capacity constraints, every district must contain
#	at least one point, hence for a district I we have
#		weight(i) + d(i, i)^2 < weight(k) + d(k, i)^2
#	=>	weight(i) < weight(k) + d(k, i)^2
#	=>	weight(i) - weight(i) < d(k, i)^2

# and if p is on the far side of i as seen from k, then
#	assign[k][p] <= assign[i][p]
# since otherwise, due to compactness, k would have to claim the
# whole region between k to p, which would include i, hence be
# contradiction that every district must contain at least one point.

# "Far side" can be formalized as
#	d(k, p) > d(i, p) and				closer to i than k
#	d(k, i) < d(k, p)					not in the middle (between i and k)

# Ideally, we could use the first observation to create a tighter big M
# constant, and the second to get rid of some unnecessary binary constraints.
# But unfortunately they don't work when fractional assignments are involved.

class SlackVoronoiCompactness:
	def has_constraints(self):
		return True

	def get_constraints(self, num_districts, num_gridpoints,
		district_point_dist, assign_binary):

		self.weight = cp.Variable(num_districts)

		constraints = []

		M = np.max(district_point_dist)**2
		g = 1	# works for >= 1000 points

		for i in range(num_districts):
			for k in range(num_districts):
				if i == k: continue
				for p in range(num_gridpoints):
					# If claimed by A, then for any other district X,
					# weight(A) - weight(X) < d(p, center_X)^2 - d(p, center_A)^2

					constraints.append(self.weight[i] - self.weight[k] - g <= district_point_dist[k][p]**2 - district_point_dist[i][p]**2 + (1 - assign_binary[i][p]) * M)

		constraints.append(self.weight >= 0)

		return constraints

	def report(self):
		print([w.value for w in self.weight])

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
	def __init__(self, compactness_constraint=NoCompactness()):
		self._name = "Hard capacitated k-means"
		self.assign = None
		self.objective_scaling_divisor = 1
		self.compactness_constraint = compactness_constraint

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
		adj_block_populations = block_populations/np.sum(block_populations)
		# tiebreak
		adj_block_populations[adj_block_populations==0] = 1e-6

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
			squared_distances_to_center += (sq_district_point_dist[district_idx] * adj_block_populations) @ self.assign[district_idx]

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
		if self.compactness_constraint.has_constraints():
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

		if self.compactness_constraint.has_constraints():
			constraints += self.compactness_constraint.get_constraints(
				num_districts, num_gridpoints, district_point_dist,
				self.assign_binary)

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
