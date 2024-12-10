'''
	1. Read census block coordinates and populations from the database.
	2. Convert census block coordinates into 3D points assuming uniform
		altitude.
	3. Determine a bounding box on the lat/long extrema of the census
		block coordinates, and convert a grid of desired granularity to
		3D points. (Essentially this does a Mercator projection and
		will probably have singularity problems near the poles, but I
		don't care at the moment.) Include the district centers we're
		going to draw a map for.
	4. Assign each census block to its closest grid point. Assign each
		grid point the total population of the census blocks closest to it.
	5. Calculate distances between each block and each center.
	6. Create the linear program.
	7. Solve it.
'''

from scipy.spatial import cKDTree, Delaunay
import numpy as np
import cvxpy as cp

import time
import pickle
import pathlib
import itertools
import subprocess

from PIL import Image

from region import Region
from problems import *
from spheregeom import *
from quant_tools import grid_dimensions

# Draws an entry from 0...num_entries proportional to weight.
def random_draw(num_entries, weight):
	norm_divisor = np.sum(weight)

	return np.random.choice(range(num_entries), p = weight/norm_divisor)

# K-means++ initial cluster selection. Let's see if these provide a better
# chance of finding a good assignment of candidate centers than plain
# Monte Carlo does.

# Distance_generator is a function that accepts an index i and returns the
# distance relative to that index, i.e. dists[k] is the distance from point k
# to point i.

def kmpp(population_counts, distance_generator, num_clusters):
	num_points = len(population_counts)
	minimal_distances = np.array([np.inf] * num_points)

	# Pick initial point
	draws = [random_draw(num_points, population_counts)]

	for i in range(1, num_clusters):
		# Update the minimal distances array to take the last added
		# point into account.
		minimal_distances = np.minimum(minimal_distances,
			distance_generator(draws[-1]))

		# Each member of the population counts as one point,
		# and each point is chosen with probability proportional to the
		# squared distance to the closest already chosen point.

		kmpp_weight = population_counts * minimal_distances ** 2

		# Draw another entry.

		new_entry = random_draw(num_points, kmpp_weight)

		# Append it to the list of chosen points.
		draws.append(new_entry)

	return np.array(draws)

def census_block_kmpp(block_data, num_clusters):
	block_populations = [block["population"] for block in block_data]
	block_latlongs = np.array(
		[[block["lat"], block["long"]] for block in block_data])

	def dist_generator(block_idx):
		block_i_center = np.array(
			[block_data[block_idx]["lat"], block_data[block_idx]["long"]])

		return haversine_center(block_i_center, block_latlongs)

	return kmpp(block_populations, dist_generator, num_clusters)

def print_claimant_array(claimants):

	for row in claimants:
		printout_string = ""
		for cell in row:
			if cell == -1:
				printout_string += "."
			else:
				printout_string += str(cell)
		print(printout_string)

def write_image(filename, pixels, aspect_ratio, region):

	height, width, error = grid_dimensions(pixels, aspect_ratio)

	# See below for why max and min is swapped for latitudes.
	img_lats = np.linspace(region.maximum_point[0], region.minimum_point[0], height)
	img_lons = np.linspace(region.minimum_point[1], region.maximum_point[1], width)

	# Create a kd tree containing the coordinates of each census block.
	block_coords = [x["coords"] for x in region.block_data]
	block_tree = cKDTree(block_coords)

	# XXX: Could use a Delaunay triangulation of the census blocks to check each
	# image point against the closest census block center's neighbors. Suppose that
	# a very large block is next to a quite small one, and the point is just inside
	# the large block. Then the small one's center might be closer even though the
	# point properly speaking belongs to the large block.

	# (We could also use Delaunay for resolution refinement. later.)

	# XXX: Also do polygon checking to account for points that are outside the
	# region (state) itself. (Or extract a polygon for the state and use polygon
	# checking against it.)

	# Creating an assignment array the way kmeans does is probably the better
	# choice here. TODO later.

	image_space_claimants = []

	for img_lat in img_lats:
		image_space_line = []
		for img_long in img_lons:
			img_coord = LLHtoECEF_latlon(img_lat, img_long, mean_radius)
			block_idx = block_tree.query(img_coord)[1]
			claimant = region.block_data[block_idx]["claimant"]
			image_space_line.append(claimant)
		image_space_claimants.append(image_space_line)

	image_space_claimants = np.array(image_space_claimants)
	claimed_num_districts = np.max(image_space_claimants)+1

	# Create some random colors.
	colors = np.random.randint(256, size = (claimed_num_districts, 3),
		dtype=np.uint8)

	# Color mixed claims grey.
	colors = np.vstack([colors, [127, 127, 127]])
	image_space_claimants[image_space_claimants==-1] = claimed_num_districts

	# And save!
	image = Image.fromarray(colors[image_space_claimants].astype(np.uint8))
	image.save(filename, "PNG")

# Get coordinates and populations. Use Colorado for now.
colorado = Region("tl_2023_08_tabblock20.dbf")

grid_points = 100

# Creates a equirectangular grid of the desired granularity.
# minimum_point is the minimum latitude and longitude of the bounding box
# surrounding the state. maximum_point ditto with the maximum.

def create_grid(grid_points, region):
	aspect_ratio = region.get_aspect_ratio()

	height, width, error = grid_dimensions(grid_points, aspect_ratio)

	# When we plot figures on screen, (0, 0) is upper left. However, latitudes are
	# greater the closer they are to the North pole. To make maps come out the right
	# way, we thus need to make earlier latitudes higher. That's why the maximum
	# and minimum points are swapped here.

	lats = np.linspace(region.maximum_point[0], region.minimum_point[0], height)
	lons = np.linspace(region.minimum_point[1], region.maximum_point[1], width)

	grid_coords = []
	grid_latlon = []

	for lat, lon in itertools.product(lats, lons):
		grid_coords.append(LLHtoECEF_latlon(lat, lon, mean_radius))
		grid_latlon.append([lat, lon])

	return np.array(grid_latlon), np.array(grid_coords), width

# max_seconds is the max number of seconds the solver should spend solving.
# It's useful for HCKM problems with a very large number of candidate points,
# see problems.py.

# TODO: Disregard it when we're refining.

def redistrict(desired_num_districts, district_indices, region=colorado,
	verbose=False, print_claimants=False, max_seconds=None):

	num_district_candidates = len(district_indices)

	grid_latlon, grid_coords, long_axis_points = create_grid(
		grid_points, region)

	district_coords = []
	district_latlon = []

	for index in district_indices:
		lat = region.block_data[index]["lat"]
		lon = region.block_data[index]["long"]
		district_coords.append(LLHtoECEF_latlon(lat, lon, mean_radius))
		district_latlon.append([lat, lon])

	district_coords = np.array(district_coords)
	district_latlon = np.array(district_latlon)

	# 4. Assign each block to its closest center.
	# This is somewhat of a hack - could do
	# https://stackoverflow.com/questions/10549402/kdtree-for-longitude-latitude
	# instead. Later.
	points_tree = cKDTree(grid_coords)
	num_gridpoints = len(grid_coords)

	grid_populations = np.zeros(num_gridpoints, np.int64)

	# TODO: I probably shouldn't be reaching into an object like this.
	# Fix later.
	for cur_block in region.block_data:
		center_idx = points_tree.query(cur_block["coords"])[1]
		cur_block["center_idx"] = center_idx
		grid_populations[cur_block["center_idx"]] += cur_block["population"]

	# 5. Create pairwise distances between centers and points.
	district_point_dist = haversine_centers(
		district_latlon, grid_latlon)

	# 6. Create the program/problem to solve.
	
	kmeans = HardCapacitatedKMeans()
	# kmeans.has_compactness_constraints = True
	# kmeans.get_compactness_constraints = HCKM_exact_compactness

	prob = kmeans.create_problem(desired_num_districts, num_gridpoints,
		grid_populations, district_point_dist)

	# ==============================================================================
	# ==============================================================================

	# cvxpy has a major bottleneck in ConeMatrixStuffing. The given parameters below
	# mitigate the problem somewhat, but it's still quite slow for larger problems
	# (try e.g. grid_points=40000.)
	# Even at default settings (grid_points=20), the old GLPK approach is
	# faster (4% faster redistrict() call); but we need to read the assignment
	# values after solving, which is pretty hard to do through a GLPK invocation.
	# I may replace this with a better solver later.

	solver_type = cp.SCIP
	scip_params = {}

	if max_seconds:
		scip_params = { "limits/time": max_seconds }

	prob.solve(verbose=verbose, ignore_dpp=True,
		canon_backend=cp.SCIPY_CANON_BACKEND, solver=solver_type,
		scip_params=scip_params)
	objective_value = prob.value / kmeans.objective_scaling_divisor

	# Get a boolean array showing which districts were chosen.
	district_choices = np.array([int(kmeans.active[x].value) == 1
		for x in range(num_district_candidates)])

	# and an array of these chosen districts.
	chosen_districts = district_indices[district_choices]

	assign_values = np.array([[var.value for var in row] for row in kmeans.assign])
	chosen_district_populations = np.sum(assign_values *
		grid_populations, axis=1)[district_choices]
	
	population_stddev = np.sqrt(np.var(chosen_district_populations))
	relative_stddev = population_stddev/np.sum(chosen_district_populations)

	if not print_claimants:
		return objective_value, chosen_districts, relative_stddev

	# If we want to refine the result, we need to invalidate some points - points
	# whose Voronoi cells might have finer detail that we don't know yet. There
	# are two types:
	#	- Points that are assigned to multiple districts (because it indicates
	#		that the point is densely populated and we need higher resolution
	#		to discern the details).
	#	- Points that neighbor points assigned to other districts (because the
	#		true, higher res border may be in the middle of a current resolution
	#		cell).

	# To do this, we first need to figure out what the points' neighbors are.
	# Let's use a Delaunay triangulation.

	# Some further experimentation indicates that there may be numerical
	# imprecision problems here. Let's see later if they pose a problem.

	grid_neighbor_source = Delaunay(grid_latlon,
		qhull_options="Qbb Qc Qz Q12 Qt")
	indptr, indices = grid_neighbor_source.vertex_neighbor_vertices

	grid_neighbors = [indices[indptr[i]:indptr[i+1]]
		for i in range(num_gridpoints)]

	# Next, we need to determine the grid points that are "stale", or
	# invalidated. First we need a mapping from grid points to districts
	# that hold them.

	# Get the values: each row is a district, each cell values[d][p] corresponds to
	# how much of grid area p's population has been allocated to that district.

	# We want to figure out which cells have been assigned to which districts, as well
	# as which cells are uncertain. The cells we can't be sure about are those that
	# have fractional values, as well as all of their direct neighbors.

	directly_certain = np.max(assign_values, axis=0) > (1 - 1e-5)

	# Which district belongs to which point. Regions with multiple districts
	# are set to -1.
	claimants = np.argmax(assign_values, axis=0)
	claimants[directly_certain == False] = -1

	# Determine stale points.

	stale_gridpoints = set()
	for i in range(num_gridpoints):
		# Points assigned to multiple districts.
		if claimants[i] == -1:
			stale_gridpoints.add(i)
			continue
		# Maybe some clever vectorization could be done here.
		for neighbor in grid_neighbors[i]:
			if claimants[neighbor] != claimants[i]:
				stale_gridpoints.add(i)

	# Next, create a higher res grid and determine which points are
	# closest/on top of a stale grid point.
	# TODO: Find a better way to do this. E.g. suppose a stale point
	# is something like a city smack dab in the middle of a large
	# set region, then the chance that our new grid gets on top of it
	# is very small. Ultimately, I need to get away from grids entirely.

	new_total_gridpoints = 150

	new_grid_latlon, new_grid_coords, new_long_axis_points = create_grid(
		new_total_gridpoints, region)

	# Find grid points that cover stale points.
	old_grid_points_covered = points_tree.query(new_grid_coords)[1]
	covers_stale = [i for i in range(new_total_gridpoints) \
		if old_grid_points_covered[i] in stale_gridpoints]
	does_new_point_cover_stale = np.array([False] * new_total_gridpoints)
	does_new_point_cover_stale[covers_stale] = True

	# Next: make redistrict take does_new_point_cover_stale as input.
	# Count the population for each district covering stale points and
	# decided points. The former go into the solver as pop[point] as
	# before; the latter is summed up to decided_pop[district] based
	# on district the decided point is closest to.
	# Then the HCKM solver population constraint should instead be
	# (3) decided[district] +
	#		for all i: (sum over p: assign[i, p] * pop[p]) <= tpop/n

	# --All of that is TODO--




	# Stuff related to outputting an image may be found below.

	# Associate each census block with the claimant for the point
	# it's closest to.

	# TODO: I probably shouldn't be reaching into an object like this either.
	# Fix later!
	for cur_block in region.block_data:
		cur_block["claimant"] = claimants[cur_block["center_idx"]]
		# Better is: get the polygon for the census block and check
		# that every vertex is in a decided block. If it is, then the
		# census block is decided, otherwise it is at least partly
		# stale. TODO?
		if cur_block["center_idx"] in stale_gridpoints:
			cur_block["certain_claimant"] = -1
		else:
			cur_block["certain_claimant"] = cur_block["claimant"]

	# See the function for why this 2D claimant array is upside down
	two_dim_claimants = np.reshape(claimants, (-1, long_axis_points))
	print_claimant_array(two_dim_claimants)

	pixels = 600**2 # e.g., total number of pixels used in output image.

	aspect_ratio = region.get_aspect_ratio()
	write_image(f"output_test_pop_{chosen_districts[0]}_points{grid_points}.png",
		pixels, aspect_ratio, region)

	return objective_value, chosen_districts, relative_stddev

# Things that need to be done if we want to refine things:
# - For each point, find the points whose Voronoi cells touch the point's.
# - If the current point is fractionally assigned, set it and all its neighbors
#		to unknown
# - If the current point has a neighbor that belongs to another district, set
#		both to unknown
# - Otherwise, assign that point to the district (the assignment is definite).

# The sticking point so far is how to robustly find every neighbor of a given
# point, even when these points may be irregularly placed, as might happen when
# refining the map.

def run(district_indices=None, region=colorado):

	if district_indices is None:
		specified_district = False
	else:
		specified_district = True

	while True:
		# NOTE: Colorado can support up to 1000 here for
		# ordinary (non-compact) HCKM...
		num_districts_to_test = 54
		desired_num_districts = 8

		# If we haven't specified a district and are instead just
		# brute-forcing stuff, use a timeout (see notes in problems.py)
		# You may need to adjust it according to your computer's
		# performance. (TODO, fix later??? Something like: it we've
		# spent more than half the time without an improvement in the
		# primal, give up? Does SCIP even support that?)

		if specified_district:
			max_seconds = None
		else:
			district_indices = census_block_kmpp(
				region.block_data, num_districts_to_test)
			max_seconds = 900

		district_indices = np.array(sorted(district_indices))
		objective_value, district_indices, relative_stddev = redistrict(
			desired_num_districts, district_indices,
			print_claimants=specified_district, verbose=True,
			max_seconds=max_seconds)

		# Printing the district indices as a list makes it easier to
		# copy and paste into Python code.
		print(f"{objective_value:.4f}, rel err: "
			f"{relative_stddev:.4f}, {list(district_indices)}")

		if specified_district:
			return

district_indices = None

# Specify a district here if you want to investigate one. Potential candidates
# include:

# Good HCKM scores:
# district_indices = [1942, 4332, 29611, 37589, 39503, 102295, 119431, 136323]
# district_indices = [5377, 25548, 29624, 45261, 52434, 73520, 90033, 112030]
# district_indices = [23255, 23766, 30428, 33463, 41185, 48967, 88287, 131743]
# district_indices = [2067, 17262, 20419, 24657, 26253, 52238, 102279, 136267]
# district_indices = [18937, 19979, 74671, 83626, 97369, 107272, 123089, 130918]
# district_indices = [1100, 60990, 64540, 65189, 84319, 84798, 88665, 90398]
# district_indices = [9316, 63572, 68116, 77836, 85977, 90872, 97054, 119145]

# Good uncapacitated scores or relative std devs:
# district_indices = [30054, 47476, 59892, 61154, 72719, 98886, 120521,134888] #(score)
# district_indices = [26905, 28861, 65810, 71157, 88867, 104151,114789,138225] #stddev

# district_indices = [1942, 4332, 29611, 37589, 39503, 102295, 119431, 136323]

run(district_indices)