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

# Get coordinates and populations. Use Colorado for now.
colorado = Region("tl_2023_08_tabblock20.dbf")

grid_points = 300

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
	# kmeans = HardCapacitatedKMeans(SwapCompactness()) # for compactness constraints

	# 7. Solve it!
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

	# We want to figure out which cells have been assigned to which districts, as
	# well as which cells are uncertain (assigned to multiple districts)

	assigned_to_only_one = np.max(assign_values, axis=0) > (1 - 1e-5)

	# Which district belongs to which point. Regions with multiple districts
	# are set to -1.
	claimants = np.argmax(assign_values, axis=0)
	claimants[assigned_to_only_one == False] = -1

	# Stuff related to outputting an image may be found below.

	# Associate each census block with the claimant for the point
	# it's closest to.

	# TODO: I probably shouldn't be reaching into an object like this either.
	# Fix later!
	assignments = []
	for cur_block in region.block_data:
		cur_block["claimant"] = claimants[cur_block["center_idx"]]
		assignments.append(claimants[cur_block["center_idx"]])

	two_dim_claimants = np.reshape(claimants, (-1, long_axis_points))
	print_claimant_array(two_dim_claimants)

	pixels = 600**2 # e.g.; total number of pixels used in output image.

	region.write_image(
		f"output_test_pop_{chosen_districts[0]}_points{grid_points}.png",
		assignments, pixels)

	return objective_value, chosen_districts, relative_stddev

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

# district_indices = [6264, 38224, 48101, 70818, 79460, 81361, 103741, 139140]
# district_indices = [9316, 63572, 68116, 77836, 85977, 90872, 97054, 119145]

# district_indices = [15406, 16255, 23793, 82245, 86791, 90245, 108701, 115819]
# district_indices = [10536, 43571, 59980, 77053, 83841, 94253, 97768, 107632]
# district_indices = [1040, 20676, 70596, 85153, 85560, 99460, 105890, 118614]

run(district_indices)