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

num_districts = 8
# Use this if you want a selection with good performance. Found by earlier
# testing.
district_indices = [23255, 23766, 30428, 33463, 41185, 48967, 88287, 131743]

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
'''

minimize sum over coordinates x,y in the region:
    sum over districts i = 1..n:
(1)        assign[i, x, y] * dist^2(centerx_i, centery_i, x, y)

subject to
(2)    for all i, x, y: 0 <= assign[i, x, y] <= 1
(3)    for all i: (sum over x, y: assign[i, x, y] * pop[x, y]) = tpop/n
(4)    for all x, y: (sum over i: assign[i, x, y]) = 1

'''

# cvxopt testing

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
	pop_constraint = int(num_districts) * (assign[district_idx] @ block_populations) == int(total_population)
	pop_constraints.append(pop_constraint)

# LP part (4)

# (4)    for all x, y: (sum over i: assign[i, x, y]) = 1
assign_constraints = []

for region_pt_idx in range(num_gridpoints):
	assign_constraint = cp.sum(assign[:,region_pt_idx]) == 1
	assign_constraints.append(assign_constraint)

constraints += pop_constraints + assign_constraints

prob = cp.Problem(cp.Minimize(squared_distances_to_center), constraints)

# cvxpy has a major bottleneck in ConeMatrixStuffing. The given parameters below
# mitigate the problem somewhat, but it's still quite slow for larger problems
# (try e.g. grid_points=40000.)
# Even at default settings (grid_points=20), the old GLPK approach is
# faster (4% faster redistrict() call); but we need to read the assignment
# values after solving, which is pretty hard to do through a GLPK invocation.
# I may replace this with a better solver later.
prob.solve(verbose=True, ignore_dpp=True,
	canon_backend=cp.SCIPY_CANON_BACKEND, solver=cp.SCIP)
objective_value = prob.value

# Get the values: each row is a district, each cell values[d][p] corresponds to
# how much of grid area p's population has been allocated to that district.

# We want to figure out which cells have been assigned to which districts, as well
# as which cells are uncertain. The cells we can't be sure about are those that
# have fractional values, as well as all of their direct neighbors.


# Interior point solvers only solve to a particular accuracy.
# TODO: Get this tolerance from the solution itself.
epsilon = 1e-15

assign_values = np.array([[var.value for var in row] for row in assign])
directly_certain = np.max(assign_values, axis=0) > (1-epsilon)

# Which district belongs to which point. Regions with multiple districts
# are set to -1.
claimants = np.argmax(assign_values, axis=0)
claimants[directly_certain == False] = -1

# Now we could comment out (a) and do something like
# print(np.reshape(claimants, (-1, long_axis_points)))
# but this will be mirrored north to south and also has strange point
# inclusions of otherwise contiguous districts. I have to find out what's
# going on using a better rendering method.

# Things that need to be done:
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