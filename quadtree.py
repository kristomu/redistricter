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

# TODO?? Mark fractionally assigned stuff as different from
# undecided, if just for the linearization, so that we can do
# nearest neighbor on the borders without overwriting fractional
# assignments?

# TODO?? Use an external distance function to determine when
# sub-regions of a cell are forced to have a particular assignment,
# and then splitting on these and calling them decided?
# E.g. if the distance is Euclidean and both top points as well
# as the bottom left one are the same district, then the top left
# sub-cell must belong to that district.
# And if the population enclosed is zero then we can get an exact
# result by just finding the curve of equal distance from both
# district centers.
# I've found out that we're going to need some pruning to make
# the method anywhere near realistic.

# It should also keep every resolved point so that, when it's
# asked to draw something, it can go off the current points
# instead of being very conservative and only outputting decided
# cells.

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

	def _midpoint(self):
		return tuple(self.upper_left +
			(self.lower_right - self.upper_left)/2)

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

			# If we don't know a cell's state wrt the bounds,
			# it might be partially or completely outside, so
			# dont' return anything.
			if self.bound_state == RQ_BOUNDS_UNKNOWN:
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
	def _update_bounds_state(self, norm_span):

		if self.bound_state != RQ_BOUNDS_UNKNOWN:
			return # No need to check

		# If our lower right value is greater than the norm span
		# in any direction, then we're partly or fully out of
		# bounds.

		# If, in addition, one or more of our *upper right* values
		# exceed the norm span, then we're entirely outside.

		# Test the latter first.

		if np.any(self.upper_left > norm_span):
			self.bound_state = RQ_BOUNDS_OUTSIDE
			self.point_state = RQP_DECIDED
			return

		if np.any(self.lower_right > norm_span):
			self.bound_state = RQ_BOUNDS_CROSSING
			return

		self.bound_state = RQ_BOUNDS_INSIDE

	# Check if we're partially out of bounds, and if so, split.
	# If we're a leaf node, completely out of bounds and don't know it,
	# set our state to out_of_bounds as well.
	def split_on_bounds(self, norm_span):

		# First determine our own bounds status if we don't know.

		if self.bound_state == RQ_BOUNDS_UNKNOWN:
			self._update_bounds_state(norm_span)

		# Then check our children. We need to do this even if we're
		# a point that's entirely inside, because our children might
		# have an unknown bound state.
		# TODO: Do this in a more elegant way by figuring out a cell's
		# bound state on initialization. But then we need to carry around
		# the bounds everywhere, and I'm not sure how to do that yet.

		if not self.leaf:
			# Check our children.
			for child in self.children:
				child.split_on_bounds(norm_span)

		if self.bound_state == RQ_BOUNDS_CROSSING:
			# So we're a cell that's partly outside. Split if we don't
			# already have children.

			if self.leaf:
				self.split()

				# Set the children's bound statuses.
				for child in self.children:
					child._update_bounds_state(norm_span)

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
				child.split_on_points(assignments)

	def get_decided_points(self):
		if self.leaf:
			if self.point_state == RQP_DECIDED and \
				self.bound_state != RQ_BOUNDS_OUTSIDE:

				return [(self._midpoint(), self.assigned_to)]
			return []
		else:
			points = []

			for child in self.children:
				points += child.get_decided_points()

			return points

	# When solving a districting problem, we need the undecided points,
	# which we want to find the assignments for; and we also need the
	# decided points to absorb the population that we've already found
	# the proper assignment for, without having to add a lot of dummy
	# variables to the MIP.

	# This function gets the decided points neighboring the ones that
	# are currently undecided. How this works in context of k-means is
	# that these "boundary points" will be closest to every point that
	# has been solved in some prior round.

	def get_neighboring_decided_points(self, undecided_point_set):
		if self.leaf:
			if self.point_state == RQP_DECIDED:
				corners = self._get_corners()

				if len(corners.intersection(undecided_point_set)) > 0:
					return set([(corner, self.assigned_to) for corner in corners])
			return set()
		else:
			output_set = set()

			for child in self.children:
				output_set = output_set.union(
					child.get_neighboring_decided_points(undecided_point_set))

			return output_set

	# Render the quadtree into a linear array (picture).
	# Perhaps do something that will fill out the -1s along the
	# boundaries as those will never really go away... not sure what
	# that would be, though.
	def _linearize(self, resolution, array):
		# By induction, the whole cell should be at least
		# a pixel wide. If we're not a leaf, check if our
		# children are at least a pixel wide. If they are,
		# recurse. Otherwise draw our values to the array.
		if not self.leaf:
			if (resolution * (self.lower_right[0]-self.upper_left[0])) > 0.5:
				for child in self.children:
					child._linearize(resolution, array)
				return


		start = np.round(resolution * self.upper_left).astype(int)
		end = np.round(resolution * self.lower_right).astype(int)

		# Y first.
		array[start[1]:end[1], start[0]:end[0]] = self.assigned_to

	def linearize(self, resolution):
		# TODO: Fill with None or -2 or something so we can
		# tell them apart from district zero if something goes
		# wrong.
		x = np.zeros([resolution, resolution])

		self._linearize(resolution, x)

		return x