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

# The quadtree uses "normalized" units, where the root cell ranges from 0 to 1 on
# both axes, with (0,0) anchored at the upper left of the natural bounding
# rectangle. This means that unless our input is a square, parts of the tree will
# be out of bounds.

# At some point I should factor this properly. (Doing so is doubly important in
# Python because entropy *will* kick your ass.) But let's get it working first.

# This gives the lower right of the bounding rectangle in normalized units.

norm_span = (nat_lower_right - nat_upper_left) / \
	(np.array(bounding_square_lr) - nat_upper_left)

def get_value(point):

	x, y = point

	if x < 0 or x > 1:
		raise IndexError("get_value: x is out of bounds")

	if y < 0 or y > 0.5:
		raise IndexError("get_value: y is out of bounds")

	# The diagonal is defined by y = x/2.
	if x >= y * 2:
		return 0		# first color

	return 1			# second color

root = RQuadtree()

def step(root):
	# Get all the undecided points. (When we start doing solving stuff, we'll
	# also have to get adjacent *decided* points to assign the known population
	# assigned to a district to. But that's for later.)

	root.split_on_bounds(norm_span)
	undecideds = root.get_undecided_points()

	# Resolve them and update the cells' statuses.
	assignments = {point: get_value(point) for point in undecideds}

	root.split_on_points(assignments)

	print(root.get_decided_points())

# Repeat step() as many times as you'd like, e.g...

for i in range(5):
	step(root)