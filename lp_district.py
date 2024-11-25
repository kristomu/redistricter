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

from problems import *

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


def print_claimant_array(claimants):
	# Because higher latitudes come later but correspond to
	# a more northernly posiiton, the array will be "upside down".
	# So print the rows in reverse order.
	# (Don't print stuff near or south of the equator with this!)
	for row in np.flip(claimants, axis=0):
		printout_string = ""
		for cell in row:
			if cell == -1:
				printout_string += "."
			else:
				printout_string += str(cell)
		print(printout_string[:-1])

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

grid_points = 100

# Get roughly the desired number of grid points at roughly the right
# aspect ratio by rounding off square roots.

long_axis_points = int(np.round(np.sqrt(grid_points) * EW_distance/NS_distance))
lat_axis_points = int(np.round(np.sqrt(grid_points) * NS_distance/EW_distance))

# TODO: Make this rectangular so that we know if we get the axes right
# later.

lats = np.linspace(minimum_point[0], maximum_point[0], lat_axis_points)
lons = np.linspace(minimum_point[1], maximum_point[1], long_axis_points)

def redistrict(desired_num_districts, district_indices, verbose=False,
	print_claimants=False):

	num_district_candidates = len(district_indices)

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

	#grid_coords += district_coords
	#grid_latlon += district_latlon

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
	district_point_dist = haversine_centers(
		district_latlon, grid_latlon)

	# 6. Create the program/problem to solve.
	
	kmeans = HardCapacitatedKMeans()
	#kmeans.get_compactness_constraints = HCKM_exact_compactness

	prob = kmeans.create_problem(desired_num_districts, num_gridpoints,
		block_populations, district_point_dist)

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

	prob.solve(verbose=verbose, ignore_dpp=True,
		canon_backend=cp.SCIPY_CANON_BACKEND, solver=solver_type)
	objective_value = prob.value / kmeans.objective_scaling_divisor

	# Get a boolean array showing which districts were chosen.
	district_choices = np.array([int(kmeans.active[x].value) == 1
		for x in range(num_district_candidates)])

	# and an array of these chosen districts.
	chosen_districts = district_indices[district_choices]

	assign_values = np.array([[var.value for var in row] for row in kmeans.assign])
	chosen_district_populations = np.sum(assign_values *
		block_populations, axis=1)[district_choices]
	
	population_stddev = np.sqrt(np.var(chosen_district_populations))
	relative_stddev = population_stddev/np.sum(chosen_district_populations)

	if not print_claimants:
		return objective_value, chosen_districts, relative_stddev

	# NOTE: The current display function will produce weird results if not all
	# districts were chosen, since it will index based on input district number,
	# not chosen district number. This may lead to large integers being output
	# in the makeshift map, which, strictly speaking, is correct but doesn't do
	# user readability much good.

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

	# See the function for why this 2D claimant array is upside down
	two_dim_claimants = np.reshape(claimants, (-1, long_axis_points))
	print_claimant_array(two_dim_claimants)

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

def run(district_indices=None):

	if district_indices is None:
		specified_district = False
	else:
		specified_district = True

	while True:
		desired_num_districts = 8
		num_districts_to_test = 54

		if not specified_district:
			district_indices = np.random.choice(range(len(block_data)),
				size=num_districts_to_test, replace=False)

		district_indices = np.array(sorted(district_indices))
		objective_value, district_indices, relative_stddev = redistrict(
			desired_num_districts, district_indices,
			print_claimants=specified_district, verbose=True)

		print(f"{objective_value:.4f}, rel err: "
			f"{relative_stddev:.4f}, {district_indices}")

		if specified_district:
			return

district_indices = None

# Specify a district here if you want to investigate one. Potential candidates
# include:

# Good HCKM scores:
# district_indices = [23255, 23766, 30428, 33463, 41185, 48967, 88287, 131743]

# Good uncapacitated scores or relative std devs:
# district_indices = [30054, 47476, 59892, 61154, 72719, 98886, 120521,134888] #(score)
# district_indices = [26905, 28861, 65810, 71157, 88867, 104151,114789,138225] #stddev

run(district_indices)