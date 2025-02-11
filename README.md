# redistricter
Redistricting program based on census data and linear optimization

## Requirements
The current version uses .zip files from the [TIGER2023 census block dataset.](https://www2.census.gov/geo/tiger/TIGER2023/TABBLOCK20/) as well as the [census integer to state names mapping](https://www2.census.gov/geo/docs/reference/codes2020/national_state2020.txt) and the [US Census Bureau state boundary files.](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html)

The Python implementation requires the [dbfread](https://dbfread.readthedocs.io/en/latest/) library, as well scipy and [CVXpy](https://www.cvxpy.org/) and scipy.

## Strategy

The redistricting algorithm takes a number of centers and finds compact
districts of equal populations, centered on these center coordinates. The
compactness measure to be optimized is the sum of squared distances from the
each census block to its associated district's center, subject to each district
having equal population.

As requiring each district to contain a whole number of census blocks may be
impossible or at least intractable, some blocks may be covered by multiple
districts. This permits either a subdivision below census block level, or the
blocks can be apportioned at random, giving some differences in population.

Since distances are well-behaved, the assignment can be sped up by first using
a coarse map and assigning grid points, then retaining the areas that must
belong to each district no matter the resolution, then repeatedly resolving the
ambiguous areas at a higher resolution.

## What has been implemented so far

lp\_district.py finds reasonable district centers using mixed-integer
programming with cvxpy. kmeans.py optimizes these to make populations as close
to equal as possible.

I tried iterative refinement in lp\_quadtree.py, but it did not work well. More
work needs to be done if it's to be viable. For now, using kmeans to enforce
the population constraint seems to work.
