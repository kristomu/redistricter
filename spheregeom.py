# Spherical geometry stuff

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# mean distance from the center of the Earth, in kilometers.
mean_radius = 6371

# https://stackoverflow.com/a/20360045
# Input coordinates are in radians.
def LLHtoECEF_rad(r_lat, r_lon, alt):
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

def LLHtoECEF_latlon(lat, lon, alt=mean_radius):
	return LLHtoECEF_rad(np.radians(lat), np.radians(lon), alt)

# https://stackoverflow.com/a/29546836
# Output is in kilometers.
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
	if len(centers_latlon) > len(other_points):
		return haversine_centers(other_points, centers_latlon).T

	each_center_iter = map(
		lambda center: haversine_center(center, other_points),
		centers_latlon)

	return np.array(list(each_center_iter))

# Generate an "optimistic distance" matrix.
# District i's optimistic distance to point p is the distance from i to
# the closest point that's closer to p than any other point. **
# Let p's assigned points be the points that, in the full resolution problem
# (no quantization) are separate but that, due to quantization, have been
# aggregated into point p.
# Then the relevance of the optimistic distance from i to p is that it is
# a lower bound on any weighted mean distance between i and any subset of
# points assigned to p.
# Thus using the optimistic distance in a hard cap problem should lower
# bound the objective value for the completely disaggregated problem,
# allowing for the early rejection of unpromising solutions.

# ** This would ordinarily be the closest point to i in p's Voronoi
# region, but scipy.spatial.Voronoi is kind of a pain when dealing with
# grid points and bounding boxes (not to mention non-convex state
# boundaries). Fortunately for us, the logic still works if we say
# "of the census blocks that have p as its nearest neighbor, the one
# that is closest to i".

# SOME TESTING LATER:
#	Well, yes, it does provide a lower bound. It's just a shame that
#	that lower bound is very loose.

# For instance, for this Colorado example:
# Distance: 3509.9626 Pop. std.dev: 29.2350, max-min: 88 for
# 	[7449, 19081, 57667, 62841, 67271, 70950, 81151, 92478]
# solving with 4000 grid points, HCKM with no compactness constraints:
#	using actual distances gives 3533.7747 (0.6% too high)
#	using optimistic distances, gives 3233.4548 (8% too low).
# with 500 points:
#	using actual distances gives 3626.7935 (3.3% too high)
#	using optimistic distances gives 2751.3964 (22% too low)

def get_optimistic_distances(district_latlon, grid_latlon, region):
	num_districts = len(district_latlon)
	num_gridpoints = len(grid_latlon)

	block_latlon = region.get_block_latlongs()

	# Just do it brute force, fix later

	block_grid_distances = haversine_centers(
		block_latlon, grid_latlon)
	district_block_dist = haversine_centers(
		district_latlon, block_latlon)
	district_point_dist = haversine_centers(
		district_latlon, grid_latlon)

	# Get nearest neighbors for all blocks.
	block_neighbors = np.argmin(block_grid_distances, axis=1)

	optimistic_dist = np.zeros((num_districts, num_gridpoints))

	for p in range(num_gridpoints):
		candidate_blocks = np.where(block_neighbors==p)
		for d in range(num_districts):
			try:
				optimistic_dist[d][p] = np.min(district_block_dist[d][candidate_blocks])
			except ValueError:
				# No block is closest to this point, so the point has pop zero, and
				# so it doesn't matter what we put here. But just to be accurate,
				# use the district-point distance.
				optimistic_dist[d][p] = district_point_dist[d][p]

	return optimistic_dist