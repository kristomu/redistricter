
# Region (state) data and operations based on it.

from parse_dbf import get_state_names, get_census_block_data

import numpy as np
from spheregeom import *

class Region:
	def __init__(self, source_dbf_file):
		self.state_names = state_names = get_state_names()
		self.block_data = get_census_block_data("tl_2023_08_tabblock20.dbf",
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

	def get_district_block_distances(self, district_indices):
		block_latlongs = np.array(
			[[block["lat"], block["long"]] for block in self.block_data])

		district_latlongs = [block_latlongs[i] for i in district_indices]

		district_block_distances = np.array([
			haversine_center(dl, block_latlongs)
			for dl in district_latlongs])

		return district_block_distances

	def get_aspect_ratio(self):
		NS_distance = haversine_np(
			self.minimum_point[0], self.minimum_point[1],
			self.maximum_point[0], self.minimum_point[1])
		EW_distance = haversine_np(
			self.minimum_point[0], self.minimum_point[1],
			self.minimum_point[0], self.maximum_point[1])

		return EW_distance/NS_distance
