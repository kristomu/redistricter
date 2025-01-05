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

	def find_enclosing_blocks(self, assignment, region):
		print("Doing image space mapping.")

		self.image_space_claimants = []

		for img_lat_idx in tqdm(range(self.height)):
			img_lat = self.img_lats[img_lat_idx]
			image_space_line = []
			for img_long in self.img_lons:
				try:
					block_idx = region.find_enclosing_block(img_lat, img_long)
					claimant = assignment[block_idx]
				except KeyError:
					claimant = -1

				image_space_line.append(claimant)
			self.image_space_claimants.append(image_space_line)

		self.image_space_claimants = np.array(self.image_space_claimants)
		claimed_num_districts = np.max(self.image_space_claimants)+1

	# Assignment is a list that, for each census block, gives what district
	# that census block belongs to.
	def write_image(self, filename, assignment, region):
		# XXX: Could use a Delaunay triangulation of the census blocks to check each
		# image point against the closest census block center's neighbors. Suppose that
		# a very large block is next to a quite small one, and the point is just inside
		# the large block. Then the small one's center might be closer even though the
		# point properly speaking belongs to the large block.

		# (We could also use Delaunay for resolution refinement. later.)

		# XXX: Also do polygon checking to account for points that are outside the
		# region (state) itself. (Or extract a polygon for the state and use polygon
		# checking against it.)

		self.find_enclosing_blocks(assignment, region)

		print("Trying to find suitable colors.")

		# We want adjacent districts to have different colors. And we want
		# the district indices to all be nonnegative, so turn mixed claims
		# or ones where we couldn't find the right district grey.
		# TODO: Don't scribble over stuff that may be used elsewhere.
		# Fix later.
		self.image_space_claimants[image_space_claimants==-1] = claimed_num_districts

		colors = self.get_colors(claimed_num_districts,
			self.image_space_claimants)

		print("Found colors.")

		# And save!
		image = Image.fromarray(colors[image_space_claimants].astype(np.uint8))
		image.save(filename, "PNG")

	# TODO: Make a "estimate_population_difference" function that takes a claimant
	# array and checks how it splits the different census blocks, thus determining
	# an average case population difference based on each census block having uniform
	# population density.
	# Then if I want to be fancy, use linear programming to determine a worst-case
	# population difference where all the population is on the wrong side of the
	# divider.