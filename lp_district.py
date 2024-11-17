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

# I need to clean this up; it's getting a bit messy what's radians,
# what's latitudes and longitudes, what corresponds to districts,
# and what corresponds to grid points.

grid_axis_points = 20		# will produce n^2 points

lats = np.linspace(minimum_point[0], maximum_point[0], grid_axis_points)
lons = np.linspace(minimum_point[1], maximum_point[1], grid_axis_points)

# Pick some census blocks as district centers
# Uncomment the fixed definition to use blocks with good properties.
def redistrict(num_districts, district_indices):

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
	# people close to the center get assigned to it. (Or do we?)

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

	# Let's do it in GMPL format first. I'll do the Ax,b format later.

	total_population = sum(block_populations)
	filename = f"redistricting_{np.random.randint(0, 2**63)}.mod"

	redist = open(filename, "w")

	# LP part (0)

	for district_idx in range(num_districts):
		for region_pt_idx in range(num_gridpoints):
			redist.write(f"var assn_d{district_idx}p{region_pt_idx} >= 0;\n")

	# LP part (1)

	objective_funct = ""

	for district_idx in range(num_districts):
		for region_pt_idx in range(num_gridpoints):
			objective_funct += f"assn_d{district_idx}p{region_pt_idx} * " + \
				f"{sq_district_point_dist[district_idx][region_pt_idx]} + "

	objective_funct = objective_funct[:-3] + ";" # remove trailing space and plus

	redist.write(f"minimize squared_dist: {objective_funct}\n")

	# LP part (2)

	for district_idx in range(num_districts):
		for region_pt_idx in range(num_gridpoints):
			redist.write(f"s.t. bound_d{district_idx}p{region_pt_idx}: " + \
				f"assn_d{district_idx}p{region_pt_idx} <= 1;\n")

	# LP part (3)
	# We move the n term to the left-hand side to limit floating point problems.

	for district_idx in range(num_districts):
		pop_constraint = ""
		for region_pt_idx in range(num_gridpoints):
			pop_constraint += f"{num_districts} * " + \
				f"assn_d{district_idx}p{region_pt_idx} * " + \
				f"{block_populations[region_pt_idx]} + "
			
		pop_constraint = pop_constraint[:-3]

		redist.write(f"s.t. pop_bound_d{district_idx}: {pop_constraint} = {total_population};\n")

	# LP part (4)

	# (4)    for all x, y: (sum over i: assign[i, x, y]) = 1
	for region_pt_idx in range(num_gridpoints):
		assign_constraint = ""
		for district_idx in range(num_districts):
			assign_constraint += f"assn_d{district_idx}p{region_pt_idx} + ";

		assign_constraint = assign_constraint[:-3]

		redist.write(f"s.t. assign_fully_p{region_pt_idx}: {assign_constraint} = 1;\n")

	# 7. Solve it.

	redist.close()

	# https://stackoverflow.com/a/707001

	solver = subprocess.Popen(["glpsol", "--math", filename],
		stdout=subprocess.PIPE)
	(output, error) = solver.communicate()
	
	pathlib.Path(filename).unlink()

	objective_value_line = [x for x in output.decode().split("\n") if len(x) > 0 and x[0] == '*'][0]
	objective_value = objective_value_line.split()[4]

	return objective_value, district_indices

	# Now suppose we have an array of assignment loads so that
	# assign[district][gridpoint] = fraction of this gridpoint belonging to that district.
	# And I want to plot them.
	# Probably I would take the lazy Plate carrée approach because inverting a map
	# projection is not my idea of a good time.

	# Perhaps https://stackoverflow.com/questions/78375583/draw-a-filled-polygon-with-fill-color-inside-the-outline

# STUB: Just do the districting over and over. The redistrict function will list
# the objective value (smaller is better).

while True:
	num_districts = 8
	# Use this if you want a selection with good performance. Found by earlier
	# testing.
	# district_indices = [23255, 23766, 30428, 33463, 41185, 48967, 88287, 131743]

	district_indices = np.random.randint(0, len(block_data), size=num_districts)
	district_indices = np.array(sorted(district_indices))
	objective_value, district_indices = redistrict(num_districts, district_indices)

	print(f"{objective_value} for {district_indices}")