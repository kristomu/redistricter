# Implement a relatively simple quadtree for the redistricter.

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

RQ_STALE = 0
RQ_UNDECIDED = 1
RQ_DECIDED = 2
RQ_OUT_OF_BOUNDS = 3
RQ_PARTLY_OUT_OF_BOUNDS = 4

class RQuadtree:
	def __init__(self, upper_left, lower_right):
		# In (x,y) number coordinates
		self.upper_left_idx = upper_left
		self.lower_right_idx = lower_right

		# These are in the order upper left, upper right,
		# lower left, lower right.
		self.children = [None] * 4

		self.state = RQ_STALE

	# Initialize as a root node.
	def __init__(self):
		self.__init__( (0, 0), (2**64, 2**64) )

	def _mid(self, a, b):
		if a > b:
			return b + (a-b)/2
		return a + (b-a)/2

	def split(self):
		# TODO: Check if we've already split; if so, raise
		# an exception.

		# The x and y coordinates are in left-middle-right or top-middle-bottom
		# order.

		x = [upper_left[0],
			 _mid(upper_left[0], lower_left[0]),
			 lower_left[0]]

		y = [upper_left[1],
			 _mid(upper_left[1], lower_left[1])
			 lower_left[1]]

		self.child[0] = RQuadtree( (x[0], y[0]), (x[1], y[1]) )
		self.child[1] = RQuadtree( (x[1], y[0]), (x[2], y[1]) )

		self.child[2] = RQuadtree( (x[0], y[1]), (x[1], y[2]) )
		self.child[2] = RQuadtree( (x[1], y[1]), (x[2], y[2]) ) 