# Implement a relatively simple quadtree, with relatively
# less simple additional behavior, for the redistricter.

# Also needs an iterator "leaves".

'''
	and get_points() something like
	(upper_left.x, upper_left.y)
	(lower_right.x, upper_left.y)
	(upper_left.x, lower_right.y)
	(lower_right.x, lower_right.y)
	if children:
		append their points?
	would probably need to consider the status of the cell
	partly out of bounds, entirely out of bounds,
	undecided, decided...
'''

'''

Each quadtree node contains the state given either by the assignment
of its current vertices (if it's a leaf node) or by the states of its
children (if it's not).

The logic is like this:
	If we're a leaf node, then each corner corresponds to a point on
	the terrain that we use the solver to assign to a district.

	If they all agree and are decisive (assign the point only to one
	district), then we know that every point inside the cell must
	belong to that district. Hence the cell is decided (definitely
	assigned to that district).

	If they don't, then it's undecided and we need to split in order
	to determine just where the border goes.

	In addition, we need "partly out of bounds" and "wholly out of
	bounds" states if the bounding region is non-square. (I'll
	refine the border of the state/region more cleanly later -
	or perhaps I could just use a point in polygon test...
	also for later.)

	We also need a stale state for nothing has been resolved yet.

The inheritance rules are:
	If something changed in one of our children and we don't know
	about it yet, then we're stale.

	If some of our children are undecided, then we're undecided.

	If all of our children are decided, then we're decided.

	If some of our children are out of bounds and some are not,
	then we're partially out of bounds.

	If all our children are out of bounds, so are we.
'''

import numpy as np

RQ_STALE = 0
RQ_UNDECIDED = 1
RQ_DECIDED = 2
RQ_OUT_OF_BOUNDS = 3
RQ_PARTLY_OUT_OF_BOUNDS = 4

class RQuadtree:
	# Initialize as a root node by default
	def __init__(self, upper_left=None, lower_right=None):
		if upper_left is None and lower_right is None:
			upper_left = (0, 0)
			lower_right = (1., 1.)

		# In (x,y) number coordinates
		self.upper_left_idx = np.array(upper_left)
		self.lower_right_idx = np.array(lower_right)

		# These are in the order upper left, upper right,
		# lower left, lower right.
		self.children = [None] * 4
		self.leaf = True

		self.state = RQ_STALE

	def _mid(self, a, b):
		if a > b:
			return b + (a-b)/2
		return a + (b-a)/2

	def get_undecided_points(self):
		if self.leaf:
			if self.state == RQ_UNDECIDED:
				return set(self.upper_left_idx, self.lower_right_idx)
			return set()

		if self.state != RQ_UNDECIDED:
			return set()
		
		joined_points = set()

		for child in self.children:
			joined_points = joined_points.union(
				child.get_undecided_points())

		return joined_points

	def split(self):
		if not self.leaf:
			raise Exception("Quadtree: trying to split a non-leaf node!")

		self.leaf = False

		# The x and y coordinates are in left-middle-right or top-middle-bottom
		# order.

		x = [self.upper_left_idx[0],
			 self._mid(self.upper_left_idx[0], self.lower_right_idx[0]),
			 self.lower_right_idx[0]]

		y = [self.upper_left_idx[1],
			 self._mid(self.upper_left_idx[1], self.lower_right_idx[1]),
			 self.lower_right_idx[1]]

		self.children[0] = RQuadtree( (x[0], y[0]), (x[1], y[1]) )
		self.children[1] = RQuadtree( (x[1], y[0]), (x[2], y[1]) )

		self.children[2] = RQuadtree( (x[0], y[1]), (x[1], y[2]) )
		self.children[3] = RQuadtree( (x[1], y[1]), (x[2], y[2]) ) 

	# Change a stale node into some other kind of node based on the
	# current tree structure, before we determine the values of the
	# nodes.

	def update_state_before_resolving(self, bounding_square_ul,
		bounding_square_lr, nat_upper_left, nat_lower_right):

		# Assume that 1 is the whole range from bounding_square_ul
		# to bounding_square_lr.

		span = (bounding_square_lr - bounding_square_ul)

		# In what direction (if any) do we extend past the natural
		# bounds in the upper (northwest) direction?
		exceeds_upper = np.logical_or(
			bounding_square_ul + span * self.upper_left_idx < nat_upper_left,
			bounding_square_ul + span * self.upper_left_idx > nat_lower_right)

		exceeds_lower = np.logical_or(
			bounding_square_ul + span * self.lower_right_idx < nat_upper_left,
			bounding_square_ul + span * self.lower_right_idx > nat_lower_right)

		if np.all(exceeds_upper) and np.all(exceeds_lower):
			self.state = RQ_OUT_OF_BOUNDS
			return

		if np.any(exceeds_upper) or np.any(exceeds_lower):
			self.state = RQ_PARTLY_OUT_OF_BOUNDS

	# Check if we're partially out of bounds, and if so, split.
	# If we're a leaf node, completely out of bounds and don't know it,
	# set our state to out_of_bounds as well.
	def split_on_bounds(self, bounding_square_ul, bounding_square_lr):
		if not self.leaf and self.state == RQ_PARTLY_OUT_OF_BOUNDS:
			for child in self.children:
				child.split_on_bounds(bounding_square_ul,
					bounding_square_lr)

		# If we're not partly out of bounds or stale, there's no
		# out-of-bounds to split.
		if self.state == RQ_DECIDED or self.state == RQ_UNDECIDED or \
			self.state == RQ_OUT_OF_BOUNDS:
			return