# Mockup concept for the quadtree.

# We create a rectangle of 2:1 aspect ratio with a diagonal going through it.
# We then define the quadtree to cover the whole rectangle; as a square, this
# gives it an out-of-bounds region.

# (I could also just do it anamorphically but whatever)

from quadtree import RQuadtree
import numpy as np

nat_upper_left = np.array((0, 0))
nat_lower_right = np.array((1, 0.5))

bounding_square_ul = nat_upper_left
bounding_square_lr = np.array([np.max(nat_lower_right), np.max(nat_lower_right)])

# The point must be in natural units, i.e. the units above, not
# the quadtree's 0..2**64 scale.

def get_value(point):

	x, y = point

	if x < 0 or x > 1:
		raise KeyError("get_value: x is out of bounds")

	if y < 0 or y > 0.5:
		raise KeyError("get_value: y is out of bounds")

	# The diagonal is defined by y = x/2.
	if x >= y * 2:
		return 0		# first color

	return 1			# second color

root = RQuadtree()

# Get all the undecided points. (When we start doing solving stuff, we'll
# also have to get adjacent *decided* points to assign the known population
# assigned to a district to. But that's for later.)

root.split_on_bounds(bounding_square_ul, bounding_square_lr,
	nat_upper_left, nat_lower_right)
undecideds = root.get_undecided_points()

# Resolve them and update the cells' statuses.
assignments = {point: get_value(point) for point in undecideds}

root.split_on_points(assignments)