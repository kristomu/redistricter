# An output image class for rendering district assignments and
# calculcating stats about them.

# It violates single responsibly somewhat because everything that
# relates to image space will go here. But better get it done than
# try for perfection...

# Currently everything is done with an equirectangular projection.
# Other projections are still possible by generating grid points
# appropriately, but not yet implemented.

from PIL import Image
import numpy as np
import itertools
from collections import defaultdict

from region import Region
from quant_tools import grid_dimensions

from tqdm import tqdm

from skimage.color import rgb2lab, deltaE_ciede2000

MIXED_OR_UNKNOWN = -1
OUT_OF_BOUNDS = -2

class GeoImage:
	def __init__(self, minimum_point, maximum_point,
		pixels, aspect_ratio):

		self.height, self.width, error = grid_dimensions(pixels, aspect_ratio)

		# When we plot figures on screen, (0, 0) is upper left. However,
		# latitudes are greater the closer they are to the North pole. To
		# make maps come out the right way, we thus need to make earlier
		# latitudes higher. That's why the maximum and minimum points are
		# swapped here.

		self.img_lats = np.linspace(
			maximum_point[0], minimum_point[0], self.height)
		self.img_lons = np.linspace(
			minimum_point[1], maximum_point[1], self.width)

		self.grid_latlon = []

		for lat, lon in itertools.product(self.img_lats, self.img_lons):
			self.grid_latlon.append([lat, lon])

		self.grid_latlon = np.array(self.grid_latlon)

		# This is the mapping of points in image space to census blocks.
		self.image_space_blocks = []

	@classmethod
	def from_region(self, pixels, region):
		return GeoImage(region.minimum_point, region.maximum_point,
			pixels, region.get_aspect_ratio())

	# Auxiliary function for image generation: find which districts are
	# adjacent to each other. Returns a set, and only works horizontally.
	def get_horiz_adjacent_districts(self, image_space_claimants):
		different = set()

		for row in image_space_claimants:
			for diff_idx in np.where(row[:-1] != row[1:])[0]:
				adjacent_districts = [row[diff_idx], row[diff_idx+1]]

				# Don't count districts that are at the edge of the
				# region as adjacent to another just because it's next
				# to an out-of-bounds area.
				if OUT_OF_BOUNDS in adjacent_districts:
					continue

				adjacent_districts = tuple(sorted(adjacent_districts))

				different.add(adjacent_districts)

		return different

	def get_adjacent_districts(self, image_space_claimants):
		return self.get_horiz_adjacent_districts(image_space_claimants) \
			| self.get_horiz_adjacent_districts(image_space_claimants.T)

	def get_colors(self, claimed_num_districts, image_space_claimants,
		iters=200):

		adjacent_districts = self.get_adjacent_districts(
			image_space_claimants)

		suitability_record = 0
		record_colors = None

		for i in tqdm(range(iters)):
			# Create some random colors.
			colors=np.random.randint(256, size=(claimed_num_districts, 3),
				dtype=np.uint8)
			lab_colors = rgb2lab(colors)

			# Find the minimum pairwise distance between two colors.
			# Inefficiently :-)
			adj_diffs = np.min([deltaE_ciede2000(lab_colors[idx_x], lab_colors[idx_y])
				for idx_x, idx_y in adjacent_districts])

			global_diffs = np.min([deltaE_ciede2000(x, y)
				for x, y in itertools.permutations(colors, 2)])

			candidate_suitability = adj_diffs * 10 + global_diffs * 20
			if candidate_suitability > suitability_record:
				record_colors = colors
				suitability_record = candidate_suitability

		return record_colors

	# Determine what census block covers each image space point.
	# This is used to draw redistricting solutions where each block is
	# either fully part of a district or not at all, and for estimating
	# district populations for solutions that cut across census blocks.
	# It's only really valid for high resolution GeoImages because I don't
	# support having multiple districts in one point yet.
	def map_census_blocks(self, region):
		print("Mapping census blocks to image space.")

		self.image_space_blocks = []

		for img_lat_idx in tqdm(range(self.height)):
			img_lat = self.img_lats[img_lat_idx]

			image_space_blocks_line = []

			for img_long in self.img_lons:
				if not region.is_in_state(img_lat, img_long):
					# Record that this pixel is outside the region's
					# boundaries.
					image_space_blocks_line.append(OUT_OF_BOUNDS)
					continue

				try:
					block_idx = region.find_enclosing_block(img_lat, img_long)
				except KeyError:
					block_idx = MIXED_OR_UNKNOWN

				image_space_blocks_line.append(block_idx)
			self.image_space_blocks.append(image_space_blocks_line)

		self.image_space_blocks = np.array(self.image_space_blocks)

		# Calculate approximate population per image space square,
		# assuming even distribution.
		self.image_space_population = np.zeros(
			self.image_space_blocks.shape)

		num_blocks = len(region.block_data)
		claimed_squares = np.unique(self.image_space_blocks, return_counts=True)

		for block_idx, block_count in np.array(claimed_squares).T:
			pop_per_square = region.block_populations[block_idx]/block_count
			self.image_space_population[self.image_space_blocks==block_idx] = \
				pop_per_square

	def get_enclosing_blocks(self, block_assignment, region):
		# If we don't have the census block-image space mapping
		# yet, get it. TODO: Support different regions, e.g.
		# by naming the originator of the mapping.
		if len(self.image_space_blocks) == 0:
			self.map_census_blocks(region)

		claimants = np.copy(self.image_space_blocks)

		# Look up the assignment of every image space point
		# that is nonnegative (thus not an error or mixed point).
		claimants[claimants >= 0] = block_assignment[
			claimants[claimants >= 0]]

		return claimants

	# Block_assignment is a list that, for each census block, gives what district
	# that census block belongs to.
	def write_image(self, filename, block_assignment, region):
		# XXX: Could use a Delaunay triangulation of the census blocks to check each
		# image point against the closest census block center's neighbors. Suppose that
		# a very large block is next to a quite small one, and the point is just inside
		# the large block. Then the small one's center might be closer even though the
		# point properly speaking belongs to the large block.

		# (We could also use Delaunay for resolution refinement. later.)

		# XXX: Also do polygon checking to account for points that are outside the
		# region (state) itself. (Or extract a polygon for the state and use polygon
		# checking against it.)

		claimants = self.get_enclosing_blocks(block_assignment, region)

		claimed_num_districts = np.max(claimants)+1

		print("Trying to find suitable colors.")

		colors = self.get_colors(claimed_num_districts,
			claimants)

		# Create a color indices array that gives the special "districts"
		# (MIXED_UNKNOWN and OUT_OF_BOUNDS) positive values, and
		# then add the colors grey and transparent respectively
		# to them.
		claimant_color_indices = claimants
		claimant_color_indices[
			claimant_color_indices==MIXED_OR_UNKNOWN] = claimed_num_districts
		claimant_color_indices[
			claimant_color_indices==OUT_OF_BOUNDS] = claimed_num_districts+1

		# Must be cast to uint8, otherwise rgb2lab goes bananas.
		# I love subtle bugs.
		colors = np.vstack([colors,
			np.array([180, 180, 180], dtype=np.uint8)])

		# Add full alpha.
		colors = np.c_[colors, 255 * np.ones(len(colors))]

		# Append a transparent color for out-of-region areas.
		colors = np.vstack([colors, [0, 0, 0, 0]])

		print("Found colors.")

		# And save!
		image = Image.fromarray(colors[claimant_color_indices].astype(np.uint8),
			'RGBA')
		image.save(filename, "PNG")

	# This estimates the population distribution based on a very simple
	# model where each census block's population is equally distributed.
	def estimate_population(self, proposed_image_space_claimants, region):

		if len(self.image_space_blocks) == 0:
			self.map_census_blocks(region)

		num_districts = np.max(proposed_image_space_claimants)+1
		num_blocks = len(region.block_data)

		populations = []

		for d in range(num_districts):
			populations.append(np.sum(self.image_space_population[
				proposed_image_space_claimants==d]))

		return np.array(populations)
