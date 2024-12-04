# Spherical geometry stuff

import numpy as np

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

def LLHtoECEF_latlon(lat, lon, alt):
	return LLHtoECEF_rad(np.radians(lat), np.radians(lon), alt)

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
