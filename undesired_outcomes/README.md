# Undesired outcomes
This directory contains districting examples that show how certain redistricting
objectives or constraints can produce strange examples, or are insufficient for
eliminating such strange examples.

Every example is for Colorado unless otherwise specified.

Currently it contains:

- unweighted\_objective\_7744\_1500pts.png : Shows that minimizing the sum of squared distances of points in a district will lead the optimizer to try to make every district equally large. With an equal-population constraint and dense areas (like cities), this produces very spindly districts that each try to carve out a piece of the city. Instead, the sum of squared distances should be between people in the district, not (census block) points.<br>
<img src="unweighted_objective_7744_1500pts.png" height=100>

- pop\_compact\_9316\_6000pts.png : The wiggly boundary of the brown district shows that even with compactness constraints enabled, districts can be nonconvex.<br>
Its objective value is 3661.9881 with 6000 grid points and the following chosen district centers: [9316, 63572, 68116, 77836, 85977, 90872, 97054, 119145]. Actual objective value may differ based on grid spacing (which I might change later).<br>
<img src="pop_compact_9316_6000pts.png" height=100>

- multiplicative\_kmeans\_15242.png : Shows that approximating population constraints by making the K-means objective function "min sum over points p: pop(p) * d(p, assigned district d)<sup>2</sup> * w<sub>d</sub>" with w being free variables, produces weird nonconvex results.<br>
While this kind of pseudo-Lagrangian *does* make it possible to resize districts so that population constraints are nearly exactly met, they also produce undesired results. Instead an objective with an additive weight: pop(p) * (d(p, assigned district d)<sup>2</sup> <b>+ w<sub>d</sub></b>) should be used.<br>
The generated districting uses the following centers: [15242, 23733, 63447, 86133, 89922, 104716, 125226, 131351].<br>
<img src="multiplicative_kmeans_15242.png" height=100>
