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

from parse_dbf import get_state_names, get_census_block_data
from scipy.spatial import cKDTree
import numpy as np
import cvxpy as cp

import time
import pathlib
import itertools
import subprocess

# https://stackoverflow.com/a/20360045
# Input coordinates are in radians.
def LLHtoECEF(r_lat, r_lon, alt):
	# see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html

	rad = np.float64(6378137.0)        # Radius of the Earth (in meters)
	f = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model
	cosLat = np.cos(r_lat)
	sinLat = np.sin(r_lat)
	FF     = (1.0-f)**2
	C      = 1/np.sqrt(cosLat**2 + FF * sinLat**2)
	S      = C * FF

	x = (rad * C + alt)*cosLat * np.cos(r_lon)
	y = (rad * C + alt)*cosLat * np.sin(r_lon)
	z = (rad * S + alt)*sinLat

	return (x, y, z)

# https://stackoverflow.com/a/29546836
def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length.    
    
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    return km

# Input is center as a (lat, long) array, as well as a list of
# other points as an array of (lat, long) arrays. It returns an
# array of distances from the given center to each of the other points.
def haversine_center(center, other_points):
	duped_center = np.repeat([center], len(other_points), axis=0)

	return haversine_np(duped_center[:,0], duped_center[:,1],
		other_points[:,0], other_points[:,1])

def haversine_centers(centers_latlon, other_points):
	each_center_iter = map(
		lambda center: haversine_center(center, other_points),
		centers_latlon)

	return np.array(list(each_center_iter))


# --- K-means optimization problems -----

# Apparently hard-capacitated K-means may fail to give compact
# areas, so I'm going to experiment with uncapacitated, possibly
# with integer programming...

def hard_capacitated_kmeans(num_districts, num_gridpoints,
	block_populations, sq_district_point_dist):

	# Should the objective function instead be assign * pop * dist^2 ???

	'''

	minimize sum over coordinates x,y in the region:
	    sum over districts i = 1..n:
	(1)        assign[i, x, y] * dist^2(centerx_i, centery_i, x, y)

	subject to
	(2)    for all i, x, y: 0 <= assign[i, x, y] <= 1
	(3)    for all i: (sum over x, y: assign[i, x, y] * pop[x, y]) <= tpop/n
	(4)    for all x, y: (sum over i: assign[i, x, y]) = 1

	'''

	assign = cp.Variable((num_districts, num_gridpoints))

	total_population = int(sum(block_populations))

	# Objective function (LP part 1)
	squared_distances_to_center = 0

	for district_idx in range(num_districts):
		squared_distances_to_center += sq_district_point_dist[district_idx] @ assign[district_idx]

	# LP part (2)
	constraints = []
	constraints.append(assign <= 1)
	constraints.append(0 <= assign)

	# LP part (3)
	# We move the n term to the left-hand side to limit floating point problems.

	pop_constraints = []

	for district_idx in range(num_districts):
		pop_constraint = int(num_districts) * (assign[district_idx] @ block_populations) <= int(total_population)
		pop_constraints.append(pop_constraint)

	# LP part (4)

	# (4)    for all x, y: (sum over i: assign[i, x, y]) = 1
	assign_constraints = []

	for region_pt_idx in range(num_gridpoints):
		assign_constraint = cp.sum(assign[:,region_pt_idx]) == 1
		assign_constraints.append(assign_constraint)

	constraints += pop_constraints + assign_constraints

	prob = cp.Problem(cp.Minimize(squared_distances_to_center), constraints)

	return assign, prob, 1

# TODO: MIP version

def uncapacitated_kmeans(num_districts, num_gridpoints,
	block_populations, sq_district_point_dist):

	'''
	minimize sum over coordinates x,y in the region:
	    sum over districts i = 1..n:
	(1)        assign[i, x, y] * pop[x, y] * dist^2(centerx_i, centery_i, x, y)

	subject to
	(2)    for all i, x, y: 0 <= assign[i, x, y] <= 1
	(3)    for all x, y: (sum over i: assign[i, x, y]) = 1
	'''

	assign = cp.Variable((num_districts, num_gridpoints))

	# Add a small value so that the objective function never multiplies squared distance
	# by a population of zero.
	adj_block_populations = block_populations * 2**8 + 1

	adj_total_pop = int(sum(adj_block_populations))

	# Objective function (LP part 1)
	squared_distances_to_center = 0

	for district_idx in range(num_districts):
		squared_distances_to_center += (adj_block_populations *
			sq_district_point_dist[district_idx]) @ assign[district_idx]

	# do some normalization just so I don't have to deal with extremely large numbers.
	norm_divisor = adj_total_pop * np.mean(sq_district_point_dist)

	# LP part (2)
	constraints = []
	constraints.append(assign <= 1)
	constraints.append(0 <= assign)

	# LP part (4)

	# (4)    for all x, y: (sum over i: assign[i, x, y]) = 1
	assign_constraints = []

	for region_pt_idx in range(num_gridpoints):
		assign_constraint = cp.sum(assign[:,region_pt_idx]) == 1
		assign_constraints.append(assign_constraint)

	constraints += assign_constraints

	prob = cp.Problem(cp.Minimize(squared_distances_to_center), constraints)

	return assign, prob, norm_divisor


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

	block["coords"] = LLHtoECEF(
		radian_coords[0], radian_coords[1], mean_radius)

	if minimum_point is None:
		minimum_point = coords
	if maximum_point is None:
		maximum_point = coords

	minimum_point = np.minimum(minimum_point, coords)
	maximum_point = np.maximum(maximum_point, coords)

# 3. Create a plate carée grid of the desired granularity.

# First determine the aspect ratio. (These are off by about 10 km for
# Colorado, find out why...) (Oh, apparently it's not actually
# rectangular.)

NS_distance = haversine_np(minimum_point[0], minimum_point[1],
	maximum_point[0], minimum_point[1])
EW_distance = haversine_np(minimum_point[0], minimum_point[1],
	minimum_point[0], maximum_point[1])

grid_points = 400

# Get roughly the desired number of grid points at roughly the right
# aspect ratio by rounding off square roots.

long_axis_points = int(np.round(np.sqrt(grid_points) * EW_distance/NS_distance))
lat_axis_points = int(np.round(np.sqrt(grid_points) * NS_distance/EW_distance))

# TODO: Make this rectangular so that we know if we get the axes right
# later.

lats = np.linspace(minimum_point[0], maximum_point[0], lat_axis_points)
lons = np.linspace(minimum_point[1], maximum_point[1], long_axis_points)

def redistrict(num_districts, district_indices, verbose=False,
	print_claimants=False):

	grid_coords = []
	grid_latlon = []

	for lat, lon in itertools.product(lats, lons):
		grid_coords.append(
			LLHtoECEF(np.radians(lat), np.radians(lon), mean_radius))
		grid_latlon.append([lat, lon])

	district_coords = []
	district_latlon = []

	for index in district_indices:
		lat = block_data[index]["lat"]
		lon = block_data[index]["long"]
		district_coords.append(LLHtoECEF(lat, lon, mean_radius))
		district_latlon.append([lat, lon])

	# We need the grid to include the district points so that
	# people close to the center get assigned to it. (Or do we?) (a)

	grid_coords += district_coords
	grid_latlon += district_latlon

	grid_coords = np.array(grid_coords)
	grid_latlon = np.array(grid_latlon)

	district_coords = np.array(district_coords)
	district_latlon = np.array(district_latlon)

	# 4. Assign each block to its closest center.
	# This is somewhat of a hack - could do
	# https://stackoverflow.com/questions/10549402/kdtree-for-longitude-latitude
	# instead. Later.
	points_tree = cKDTree(grid_coords)
	num_gridpoints = len(grid_coords)

	block_populations = np.zeros(num_gridpoints, np.int64)

	for block in block_data:
		block["center"] = points_tree.query(block["coords"])[1]
		block_populations[block["center"]] += block["population"]

	# 5. Create pairwise distances between centers and points.
	# We need squared distances because the objective is to minimize the sum of
	# squared distances - we want a k-means generalization, not a k-medians.
	sq_district_point_dist = haversine_centers(
		district_latlon, grid_latlon)**2

	# 6. Create the linear program.

	# cvxopt testing

	assign, prob, norm_divisor = uncapacitated_kmeans(num_districts,
		num_gridpoints, block_populations, sq_district_point_dist)

	# ==============================================================================
	# ==============================================================================

	# cvxpy has a major bottleneck in ConeMatrixStuffing. The given parameters below
	# mitigate the problem somewhat, but it's still quite slow for larger problems
	# (try e.g. grid_points=40000.)
	# Even at default settings (grid_points=20), the old GLPK approach is
	# faster (4% faster redistrict() call); but we need to read the assignment
	# values after solving, which is pretty hard to do through a GLPK invocation.
	# I may replace this with a better solver later.

	# Use the interior point solver unless we need exact values.
	# If we're going to print out what district is assigned what grid points,
	# we're going to need exact values. (TODO: Fix this later by updating the
	# tolerance...)

	#Uncapacitated gives an unbounded outcome if I use Clarabel, figure out why
	#later...
	#solver_type = cp.CLARABEL
	solver_type = cp.SCIP

	prob.solve(verbose=verbose, ignore_dpp=True,
		canon_backend=cp.SCIPY_CANON_BACKEND, solver=solver_type)
	objective_value = prob.value / norm_divisor

	assign_values = np.array([[var.value for var in row] for row in assign])
	district_populations = np.sum(assign_values * block_populations, axis=1)
	population_stddev = np.sqrt(np.var(district_populations))
	relative_stddev = population_stddev/np.sum(district_populations)

	if not print_claimants:
		return objective_value, district_indices, relative_stddev

	# Get the values: each row is a district, each cell values[d][p] corresponds to
	# how much of grid area p's population has been allocated to that district.

	# We want to figure out which cells have been assigned to which districts, as well
	# as which cells are uncertain. The cells we can't be sure about are those that
	# have fractional values, as well as all of their direct neighbors.


	# Interior point solvers only solve to a particular accuracy.
	# TODO: Get this tolerance from the solution itself.
	solver_eps = 1e-10

	directly_certain = np.max(assign_values, axis=0) > (1-solver_eps)

	# Which district belongs to which point. Regions with multiple districts
	# are set to -1.
	claimants = np.argmax(assign_values, axis=0)
	claimants[directly_certain == False] = -1

	# For some reason, this is mirrored north to south. TODO: Find out why.
	print(np.reshape(claimants, (-1, long_axis_points)))

	return objective_value, district_indices

# Things that need to be done if we want to refine things:
# - For each point, find the points whose Voronoi cells touch the point's.
# - If the current point is fractionally assigned, set it and all its neighbors
#		to unknown
# - If the current point has a neighbor that belongs to another district, set
#		both to unknown
# - Otherwise, assign that point to the district (the assignment is definite).

# Then rerun on some grid over the unknown points.
# But I'll do that *after* I've got rendering working, because my quick and dirty
# tests seem to suggest something pretty bizarre is going on. Printing claimant
# arrays gives a district containing either both of Denver and Fort Collins or
# both of Denver and Colorado Springs; and either configuration would drastically
# exceed the population limit.

# It's possible that the linear program doesn't even guarantee contiguity:
# consider a high density city near the middle of a district, with all other
# district centers far away. Could it be that claiming only part of the city and
# then a bunch of surrounding land would decrease the penalty for the other
# districts so much that it's worth it? I'll have to investigate.

# https://www.tandfonline.com/doi/pdf/10.3846/1648-4142.2009.24.274-282 gives
# an O((dn)^2) approach to ensuring compactness:
# for each district pair d_1, d_2:
#   for each point j in d_1, s in d_2:
#       dist(j, center of d_1) + dist(s, center of d_2) <=
#           dist(j, center of d_2) + dist(s, center of d_1)
# i.e. we must not be able to swap two points belonging to different districts
# with the distance improving. (This is passed by uncapacitated facility
# location/k-means). Unfortunately this is completely impractical: e.g.
# 400 grid points, 8 districts = 10 million constraints. In addition, it
# doesn't seem to be implementable, even in theory, without using MIP.

while True:
	num_districts = 8
	# Use this if you want a selection with good performance. Found by earlier
	# testing.
	# district_indices = [23255, 23766, 30428, 33463, 41185, 48967, 88287, 131743]
	# or this: (from uncapacitated)
	# district_indices = [2995, 42888, 74000, 106266, 133597, 136444, 136963, 137608]

	district_indices = np.random.randint(0, len(block_data), size=num_districts)
	district_indices = np.array(sorted(district_indices))
	objective_value, district_indices, relative_stddev = redistrict(
		num_districts, district_indices, print_claimants=False)

	print(f"{objective_value:.4f}, rel err: {relative_stddev:.4f}, {district_indices}")