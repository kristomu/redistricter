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

# These are the vertex states. Undecided means we don't know,
# or that they don't all agree, or that at least one of the
# points is a fractional assignment, and decided means the
# vertices are all assigned to the same district, and fully so.

RQP_UNDECIDED = 0
RQP_DECIDED = 1

# These are the bound states. Unknown means that no evaluation
# has been done yet. Inside means that every vertex is inside
# or on the edge of the bounding region. Outside means that
# every vertex is outside. Crossing means that some
# are inside (or on the edge) and some are outside.

RQ_BOUNDS_UNKNOWN = 0
RQ_BOUNDS_INSIDE = 1
RQ_BOUNDS_OUTSIDE = 2
RQ_BOUNDS_CROSSING = 3

# Cells that are entirely out of bounds are decided, because in
# a way we know the status of all their points - they don't
# belong to any district.

class RQuadtree:
	# Initialize as a root node by default
	def __init__(self, upper_left=None, lower_right=None):
		if upper_left is None and lower_right is None:
			upper_left = (0, 0)
			lower_right = (1., 1.)

		# In (x,y) number coordinates
		self.upper_left = np.array(upper_left)
		self.lower_right = np.array(lower_right)

		# These are in the order upper left, upper right,
		# lower left, lower right.
		self.children = [None] * 4
		self.leaf = True

		self.point_state = RQP_UNDECIDED
		self.bound_state = RQ_BOUNDS_UNKNOWN

		# District that this cell is assigned to, if decided.
		# If it's -1 yet decided, that means that it's out of
		# bounds.
		self.assigned_to = -1

	def _mid(self, a, b):
		if a > b:
			return b + (a-b)/2
		return a + (b-a)/2

	def _get_corners(self):
		return set([
			(self.upper_left[0], self.upper_left[1]),	# upper left
			(self.lower_right[0], self.upper_left[1]),	# upper right
			(self.upper_left[0], self.lower_right[1]),	# lower left
			(self.lower_right[0], self.lower_right[1])])# lower right

	# With compactness constraints, a district can never be fully
	# enclosed in another. So we could abort as soon as we see that
	# we're decided, because that would imply that everything inside
	# our bounds are also decided. But I can afford the cycles, so
	# why not?

	# (I'm not sure if it would be a good idea to prune in that way
	# anyway.)

	# We only want the leaf nodes' corners because higher nodes may
	# remain undecided even when they're fully resolved. E.g. the root
	# node would cover a region (state) with multiple districts, and
	# it's only natural for its corners to belong to different
	# districts. That doesn't mean that these corner nodes need to be
	# resolved over and over again; only the leaf nodes that are right
	# on the boundary needs to do this.

	def get_undecided_points(self):
		if self.leaf:
			# If the cell is crossing a boundary, we don't want to
			# evaluate it; instead we better just wait until it's
			# been split into areas that are fully inside.
			if self.bound_state == RQ_BOUNDS_CROSSING:
				return set()

			if self.point_state == RQP_UNDECIDED:
				return self._get_corners()
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

		x = [self.upper_left[0],
			 self._mid(self.upper_left[0], self.lower_right[0]),
			 self.lower_right[0]]

		y = [self.upper_left[1],
			 self._mid(self.upper_left[1], self.lower_right[1]),
			 self.lower_right[1]]

		self.children[0] = RQuadtree( (x[0], y[0]), (x[1], y[1]) )
		self.children[1] = RQuadtree( (x[1], y[0]), (x[2], y[1]) )

		self.children[2] = RQuadtree( (x[0], y[1]), (x[1], y[2]) )
		self.children[3] = RQuadtree( (x[1], y[1]), (x[2], y[2]) ) 

	# Determine the current node and children's bound state.
	def _update_bounds_state(self, bounding_square_ul,
		bounding_square_lr, nat_upper_left, nat_lower_right):

		if self.bound_state != RQ_BOUNDS_UNKNOWN:
			return # No need to check

		# Assume that 1 is the whole range from bounding_square_ul
		# to bounding_square_lr.

		span = (bounding_square_lr - bounding_square_ul)

		# In what direction (if any) do we extend past the natural
		# bounds in the upper (northwest) direction?
		exceeds_upper = np.logical_or(
			bounding_square_ul + span * self.upper_left < nat_upper_left,
			bounding_square_ul + span * self.upper_left > nat_lower_right)

		exceeds_lower = np.logical_or(
			bounding_square_ul + span * self.lower_right < nat_upper_left,
			bounding_square_ul + span * self.lower_right > nat_lower_right)

		if np.all(exceeds_upper) and np.all(exceeds_lower):
			self.bound_state = RQ_BOUNDS_OUTSIDE
			self.point_state = RQP_DECIDED
			return

		if np.any(exceeds_upper) or np.any(exceeds_lower):
			self.bound_state = RQ_BOUNDS_CROSSING
			return

		self.bound_state = RQ_BOUNDS_INSIDE

	# Check if we're partially out of bounds, and if so, split.
	# If we're a leaf node, completely out of bounds and don't know it,
	# set our state to out_of_bounds as well.
	def split_on_bounds(self, bounding_square_ul,
		bounding_square_lr, nat_upper_left, nat_lower_right):

		# First determine our own bounds status if we don't know.

		if self.bound_state == RQ_BOUNDS_UNKNOWN:
			self._update_bounds_state(bounding_square_ul,
				bounding_square_lr, nat_upper_left, nat_lower_right)

		if self.bound_state != RQ_BOUNDS_CROSSING:
			return

		# So we're a cell that's partly outside. Split if we don't
		# already have children.

		if self.leaf:
			self.split()

			# Set the children's bound statuses.
			for child in self.children:
				child._update_bounds_state(bounding_square_ul,
					bounding_square_lr, nat_upper_left, nat_lower_right)
		else:
			# Otherwise check our children.
			for child in self.children:
				child.split_on_bounds(bounding_square_ul,
					bounding_square_lr, nat_upper_left, nat_lower_right)

	# Assignments is a dictionary where the points are keys and
	# the district assignments are values. An assignment of -1 means
	# that we have a fractional assignment at that point.
	def split_on_points(self, assignments):
		# Check our own corners - but not if we're partially or
		# completely out of bounds.
		try:
			corner_assignments = np.array([assignments[p] for p in \
				self._get_corners()])
			first = corner_assignments[0]

			# Are they all equal and not fractional?
			if first != -1 and (corner_assignments == first).all():
				# Then we're decided
				self.point_state = RQP_DECIDED
				self.assigned_to = first
			else:
				# Otherwise we're undecided
				self.point_state = RQP_UNDECIDED
		except KeyError:
			# Not all points were tested; if so, no problem.
			pass

		# Split if we're undecided and a leaf node. If we're
		# not a leaf node, recurse to our children.

		if self.leaf:
			if self.point_state == RQP_UNDECIDED:
				self.split()
		else:
			for child in self.children:
				child.update_points_state(assignments)