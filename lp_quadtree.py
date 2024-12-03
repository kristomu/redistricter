# Trying to reimplement the LP stuff to use quadtrees.

from parse_dbf import get_state_names, get_census_block_data
from scipy.spatial import cKDTree, Delaunay
import numpy as np
import cvxpy as cp

import time
import pickle
import pathlib
import itertools
import subprocess

from PIL import Image

from problems import *
from spheregeom import *
from quant_tools import grid_dimensions

from quadtree import RQuadtree

# Normalized coordinates to natural ones
def to_natural(normalized, minimum_point, maximum_point):
	return minimum_point + np.array(normalized) * (maximum_point-minimum_point)

# 1. Get coordinates and populations
#	  Use Colorado for now.
state_names = get_state_names()
block_data = get_census_block_data("tl_2023_08_tabblock20.dbf",
	state_names)

# 2. Convert to 3D points with fixed altitude; and 3. create a bounding box
mean_radius = 6371 # mean distance from the center of the Earth

# Bounding box coordinates are in latitude/longitude, not radian format
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

# Some initial variables just to check how things are going.

# This district choice still produces pretty wedge-shaped districts, but
# fix later.
district_indices = [1942, 4332, 29611, 37589, 39503, 102295, 119431, 136323]

# New code below!

bounding_square_ul = minimum_point
bounding_square_lr = minimum_point + np.array([
	np.max(maximum_point-minimum_point),
	np.max(maximum_point-minimum_point)])

norm_span = (maximum_point - minimum_point) / \
	(np.array(bounding_square_lr) - minimum_point)

# QT
root = RQuadtree()

district_coords = []
district_latlon = []

for index in district_indices:
	lat = block_data[index]["lat"]
	lon = block_data[index]["long"]
	district_coords.append(LLHtoECEF_latlon(lat, lon, mean_radius))
	district_latlon.append([lat, lon])

district_coords = np.array(district_coords)
district_latlon = np.array(district_latlon)

for i in range(4):
	print("Iteration", i)
	# QT
	# TODO: Fix quadtree bug where incredibly narrow rectangles
	# make undecideds return empty because every cell is either
	# CROSSING or OUTSIDE.
	root.split_on_bounds(norm_span)
	undecideds = root.get_undecided_points()
	decided_neighbors = root.get_neighboring_decided_points(
		undecideds)

	print(f"Iter {i}\tUndecided points: {len(undecideds)}, Decided neighbors: {len(decided_neighbors)}")

	# Insert the undecided points into a k-d tree to
	# calculate population numbers.
	# We should use 3D coords here but eh... let's get it working first.

	# I'll do something very dirty here: treat the decided neighbors as
	# if they were undecided. They shouldn't change; then I just pass
	# the values for the actual undecideds back to the tree.
	# Ugly? Yes. Fix later :-P

	total_points = list(undecideds) + ([coord_assign[0] \
		for coord_assign in decided_neighbors])
	total_points = list(set(total_points))

	coords_to_norm = {}
	tree_latlon = []
	for norm_point in total_points:
		latlong = to_natural(norm_point, minimum_point, maximum_point)
		tree_latlon.append(latlong)

		euclidean = LLHtoECEF_latlon(latlong[0], latlong[1], mean_radius)
		coords_to_norm[euclidean] = norm_point

	tree_latlon = np.array(tree_latlon)

	point_coords = np.array(list(coords_to_norm.keys()))

	points_tree = cKDTree(point_coords)
	num_qpoints = len(point_coords)

	tree_populations = np.zeros(num_qpoints, np.int64)

	for cur_block in block_data:
		center_idx = points_tree.query(cur_block["coords"])[1]
		cur_block["center_idx"] = center_idx
		tree_populations[cur_block["center_idx"]] += cur_block["population"]

	district_point_dist = haversine_centers(district_latlon,
		tree_latlon)

	kmeans = HardCapacitatedKMeans()
	#kmeans.has_compactness_constraints = True
	#kmeans.get_compactness_constraints = HCKM_exact_compactness

	desired_num_districts = 8

	prob = kmeans.create_problem(desired_num_districts, num_qpoints,
		tree_populations, district_point_dist)
	prob.solve(verbose=True, ignore_dpp=True,
		canon_backend=cp.SCIPY_CANON_BACKEND, solver=cp.SCIP,
		scip_params={})

	assign_values = np.array([[var.value for var in row] for row in kmeans.assign])
	directly_certain = np.max(assign_values, axis=0) > (1 - 1e-5)

	claimants = np.argmax(assign_values, axis=0)
	claimants[directly_certain == False] = -1

	resolved_points = {}

	for i in range(len(total_points)):
		if not total_points[i] in undecideds:
			continue
		resolved_points[total_points[i]] = claimants[i]

	root.split_on_points(resolved_points)