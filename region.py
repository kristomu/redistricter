
# Region (state) data and operations based on it.

from scipy.spatial import cKDTree, Delaunay
import matplotlib.path as mpltPath # https://stackoverflow.com/questions/36399381/
import numpy as np

from parse_shape import get_state_names, get_census_block_data
from quant_tools import grid_dimensions

from spheregeom import *

from PIL import Image
from itertools import permutations
from tqdm import tqdm

class Region:
	def __init__(self, source_shapefile):
		self.state_names = state_names = get_state_names()
		self.block_data = get_census_block_data(source_shapefile,
			self.state_names)

		# Determine the bounding box for the region, and augment
		# block data with coordinates.
		self.minimum_point = None
		self.maximum_point = None

		for block in self.block_data:
			coords = np.array([block["lat"], block["long"]])

			block["coords"] = LLHtoECEF_latlon(*coords)

			if self.minimum_point is None:
				self.minimum_point = coords
							    
			if self.maximum_point is None:
				self.maximum_point = coords
										    
			self.minimum_point = np.minimum(self.minimum_point, coords)
			self.maximum_point = np.maximum(self.maximum_point, coords)

		self.block_populations = np.array(
			[block["population"] for block in self.block_data])
		self.total_population = np.sum(self.block_populations)

		block_coords = [x["coords"] for x in self.block_data]
		self.block_tree = cKDTree(block_coords)

	def get_block_latlongs(self):
		return np.array(
			[[block["lat"], block["long"]] for block in self.block_data])

	def get_district_block_distances(self, district_indices):
		block_latlongs = self.get_block_latlongs()

		district_latlongs = [block_latlongs[i] for i in district_indices]

		district_block_distances = haversine_centers(
			district_latlongs, block_latlongs)

		return district_block_distances

	def get_aspect_ratio(self):
		NS_distance = haversine_np(
			self.minimum_point[0], self.minimum_point[1],
			self.maximum_point[0], self.minimum_point[1])
		EW_distance = haversine_np(
			self.minimum_point[0], self.minimum_point[1],
			self.minimum_point[0], self.maximum_point[1])

		return EW_distance/NS_distance

	# Determine if a point is inside a bounding polygon of the given
	# census block.
	def is_in_block(self, lat, lon, block_idx):
		for boundary in self.block_data[block_idx]["boundaries"]:
			boundary_poly = mpltPath.Path(boundary)
			if boundary_poly.contains_points([[lat, lon]]):
				return True

		return False

	# Find the census block index that the given point is inside.
	# Raises a key error if there is none. Points on the exact boundary
	# will return the district whose center is closest to the given point
	# by Euclidean distance.
	def find_enclosing_block(self, lat, lon):
		query_point_coord = LLHtoECEF_latlon(lat, lon, mean_radius)

		for neighbor_idx in self.block_tree.query(query_point_coord, k=10)[1]:
			if self.is_in_block(lat, lon, neighbor_idx):
				return neighbor_idx

		raise KeyError("Could not find a census block for this point.")

	# I'm not sure if this belongs here, but let's keep it here for now.
	# If I'm going to keep it here, there's also the matter of it
	# reimplementing create_grid from lp_district.py. TODO.
	def write_image(self, filename, assignment, pixels):
		aspect_ratio = self.get_aspect_ratio()
		height, width, error = grid_dimensions(pixels, aspect_ratio)

		# When we plot figures on screen, (0, 0) is upper left. However,
		# latitudes are greater the closer they are to the North pole. To
		# make maps come out the right way, we thus need to make earlier
		# latitudes higher. That's why the maximum and minimum points are
		# swapped here.

		img_lats = np.linspace(self.maximum_point[0], self.minimum_point[0], height)
		img_lons = np.linspace(self.minimum_point[1], self.maximum_point[1], width)

		# XXX: Could use a Delaunay triangulation of the census blocks to check each
		# image point against the closest census block center's neighbors. Suppose that
		# a very large block is next to a quite small one, and the point is just inside
		# the large block. Then the small one's center might be closer even though the
		# point properly speaking belongs to the large block.

		# (We could also use Delaunay for resolution refinement. later.)

		# XXX: Also do polygon checking to account for points that are outside the
		# region (state) itself. (Or extract a polygon for the state and use polygon
		# checking against it.)

		image_space_claimants = []

		print("Doing image space mapping.")

		for img_lat_idx in tqdm(range(height)):
			img_lat = img_lats[img_lat_idx]
			image_space_line = []
			for img_long in img_lons:
				try:
					block_idx = self.find_enclosing_block(img_lat, img_long)
					claimant = assignment[block_idx]
				except KeyError:
					# The point doesn't map to a census block.
					# This shouldn't happen, but currently it does, so
					# let's just paper over it.
					# TODO: Find out why.
					img_coord = LLHtoECEF_latlon(img_lat, img_long, mean_radius)
					block_idx = self.block_tree.query(img_coord)[1]
					claimant = assignment[block_idx]
					
				image_space_line.append(claimant)
			image_space_claimants.append(image_space_line)

		image_space_claimants = np.array(image_space_claimants)
		claimed_num_districts = np.max(image_space_claimants)+1

		suitable = False

		print("Trying to find suitable colors.")

		while not suitable:
			# Create some random colors.
			colors = np.random.randint(256, size = (claimed_num_districts, 3),
				dtype=np.uint8)

			# Find the minimum pairwise distance between two colors.
			# Inefficiently :-)
			diffs = np.min([np.std(x-y) for x, y in permutations(colors, 2)])

			suitable = diffs > 8

		print("OK")

		# Color mixed claims grey.
		colors = np.vstack([colors, [127, 127, 127]])
		image_space_claimants[image_space_claimants==-1] = claimed_num_districts

		# And save!
		image = Image.fromarray(colors[image_space_claimants].astype(np.uint8))
		image.save(filename, "PNG")
