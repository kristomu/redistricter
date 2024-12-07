# Undesired outcomes
This directory contains districting examples that show how certain redistricting
objectives or constraints can produce strange examples, or are insufficient for
eliminating such strange examples.

Every example is for Colorado unless otherwise specified.

Currently it contains:

- unweighted\_objective\_7744\_1500pts.png : Shows that minimizing the sum of squared distances of points in a district will lead the optimizer to try to make every district equally large. With an equal-population constraint and dense areas (like cities), this produces very spindly districts that each try to carve out a piece of the city. Instead, the sum of squared distances should be between people in the district, not (census block) points.
