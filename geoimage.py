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

		# This is the image space assignment of districts to points,
		# for the assignment method where every census block either has
		# to be completely inside or outside.
		self.image_space_claimants = []

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
				adjacent_districts = tuple(sorted(adjacent_districts))

				different.add(adjacent_districts)

		return different

	def get_adjacent_districts(self, image_space_claimants):
		return self.get_horiz_adjacent_districts(image_space_claimants) \
			| self.get_horiz_adjacent_districts(image_space_claimants.T)

	def get_colors(self, claimed_num_districts, image_space_claimants):
		adjacent_districts = self.get_adjacent_districts(
			image_space_claimants)

		suitable = False

		while not suitable:
			# Create some random colors, and add a grey color for the
			# mixed claims.
			colors = np.random.randint(256, size = (claimed_num_districts, 3),
				dtype=np.uint8)
			# Must be cast to uint8, otherwise rgb2lab goes bananas.
			# I love subtle bugs.
			colors = np.vstack([colors,
				np.array([180, 180, 180], dtype=np.uint8)])
			lab_colors = rgb2lab(colors)

			# Find the minimum pairwise distance between two colors.
			# Inefficiently :-)
			adj_diffs = np.min([deltaE_ciede2000(lab_colors[idx_x], lab_colors[idx_y])
				for idx_x, idx_y in adjacent_districts])

			global_diffs = np.min([deltaE_ciede2000(x, y)
				for x, y in itertools.permutations(colors, 2)])

			suitable = adj_diffs > 20 and global_diffs > 10

		return colors

	def find_enclosing_blocks(self, block_assignment, region):
		print("Doing image space mapping.")

		self.image_space_claimants = []
		self.image_space_blocks = []

		for img_lat_idx in tqdm(range(self.height)):
			img_lat = self.img_lats[img_lat_idx]
			image_space_line = []
			image_space_blocks_line = []
			for img_long in self.img_lons:
				try:
					block_idx = region.find_enclosing_block(img_lat, img_long)
					claimant = block_assignment[block_idx]
				except KeyError:
					claimant = -1
					block_idx = -1

				image_space_line.append(claimant)
				image_space_blocks_line.append(block_idx)
			self.image_space_claimants.append(image_space_line)
			self.image_space_blocks.append(image_space_blocks_line)

		self.image_space_claimants = np.array(self.image_space_claimants)

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

		self.find_enclosing_blocks(block_assignment, region)

		# TODO: Find out why this is 9, not 8, for Colorado...
		claimed_num_districts = np.max(self.image_space_claimants)+1

		print("Trying to find suitable colors.")

		# We want adjacent districts to have different colors, and we want
		# the district indices to all be nonnegative, so make a copy of
		# image space claimants where mixed claims are labeled with a
		# positive value, so we can easily assign a color to it.
		claimant_color_indices = self.image_space_claimants
		claimant_color_indices[claimant_color_indices==-1] = claimed_num_districts

		colors = self.get_colors(claimed_num_districts,
			claimant_color_indices)

		print("Found colors.")

		# And save!
		image = Image.fromarray(colors[claimant_color_indices].astype(np.uint8))
		image.save(filename, "PNG")

	# This estimates the population distribution based on a very simple
	# model where each census block's population is equally distributed.
	def estimate_population(self, proposed_image_space_claimants, region):

		if len(self.image_space_claimants) == 0:
			# TODO: Actually find enclosing blocks if it hasn't been done
			# yet. But then we need the block assignments, which is going
			# to be somewhat of a pain to pass around.
			raise Exception("Missing census block assignment.")

		num_districts = np.max(self.image_space_claimants)+1
		num_blocks = len(region.block_data)

		# We now count the number of image space points that each census block
		# gives to each district as an approximation of relative volume. Since
		# the image_space_claimants list lets us know the district any image
		# point belongs to, and the image_space_blocks array lets us know
		# what census block it is, this is just a matter of looking up stuff.

		# TODO: Handle mixed claims. Not sure how I'm going to do that,
		# though. Or just say outright that we don't support them.

		points_assigned_to_block = np.zeros((num_blocks, num_districts))

		for img_lat_idx in range(self.height):
			for img_lon_idx in range(self.width):
				block_idx = self.image_space_blocks[img_lat_idx][img_lon_idx]
				proposed_district = proposed_image_space_claimants[
					img_lat_idx][img_lon_idx]
				# Discard points that are out of bounds.
				# This also works as a hack for mixed claims and where we
				# couldn't figure out what block belonged at the given point.
				if proposed_district == -1:
					continue

				points_assigned_to_block[block_idx][proposed_district] += 1

		# Normalize the relative volume for each census block.
		relative_block_volume =  points_assigned_to_block / \
			(points_assigned_to_block.sum(axis=1, keepdims=True) + 1e-6)

		pops = (relative_block_volume.T * region.block_populations).T

		return pops.sum(axis=0)