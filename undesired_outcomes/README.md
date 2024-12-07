# Undesired outcomes
This directory contains districting examples that show how certain redistricting
objectives or constraints can produce strange examples, or are insufficient for
eliminating such strange examples.

Every example is for Colorado unless otherwise specified.

Currently it contains:

- unweighted\_objective\_7744\_1500pts.png : Shows that minimizing the sum of squared distances of points in a district will lead the optimizer to try to make every district equally large. With an equal-population constraint and dense areas (like cities), this produces very spindly districts that each try to carve out a piece of the city. Instead, the sum of squared distances should be between people in the district, not (census block) points.

- pop\_compact\_9316\_6000pts.png : The brown district shows that even with compactness constraints enabled, districts can be nonconvex.<br>
Its objective value is 3661.9881 with 6000 grid points and the following chosen district centers: [9316, 63572, 68116, 77836, 85977, 90872, 97054, 119145]. Actual objective value may differ based on grid spacing (which I might change later).
