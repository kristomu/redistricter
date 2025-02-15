'''
Parses TABBLOCK20 census block shapefiles, outputting a list of dicts where
each dict contains the state name, Geo ID, internal latitude,
internal longitude, and population count.
'''

import csv
import numpy as np
import shapefile
import pickle

def get_state_names(state_file="national_state2020.txt"):
	state_csv = csv.DictReader(
		open(state_file, "r"), delimiter="|")

	state_names = {}

	for state_row in state_csv:
		state_names[int(state_row["STATEFP"])] = (state_row["STATE"], state_row["STATE_NAME"])
		state_names[state_row["STATE"]] = state_row["STATE_NAME"]
		state_names[state_row["STATE_NAME"]] = state_row["STATE"]

	return state_names

# The boundary polygons for states and census blocks are lists that
# descend possibly an arbitrary depth before containing a list of tuples
# that defines the boundary of one of the components of the state or
# census block. This function recurses down such a boundary list and returns
# a list of lists of tuples.
# I'm not sure what the nested list hierarchy actually signifies,
# but this should work.
def flatten_boundary_list(boundary_list):
	output = []
	if len(boundary_list) == 0: return []

	if isinstance(boundary_list[0], tuple):
		return [boundary_list]

	for sublist in boundary_list:
		output += flatten_boundary_list(sublist)

	return output

def get_census_block_data(sf_filename, state_names):
	# Open the file if it's a string.
	sf = shapefile.Reader(sf_filename)

	data_list = []

	# First get the offsets for each label type.
	field_name = {}

	geo_dict = sf.shapeRecords().__geo_interface__

	if geo_dict["type"] != "FeatureCollection":
		raise ValueError("Shapefile is not a feature collection")

	for record_and_geometry in geo_dict["features"]:
		record = record_and_geometry["properties"]
		boundaries_longlat = record_and_geometry["geometry"]["coordinates"]

		# We use latlong by convention, so reverse the boundary points'
		# order. There may be more than one polygon (noncontiguous census blocks?
		# not sure what's going on).
		boundaries = []
		for boundary_longlat in flatten_boundary_list(boundaries_longlat):
			boundaries.append(np.array([[lat, lon] for lon,lat in boundary_longlat]))

		state_name = state_names[int(record["STATEFP20"])][1]
		geo_id = record["GEOID20"]
		lat = float(record["INTPTLAT20"])
		lon = float(record["INTPTLON20"])
		pop = int(record["POP20"])
		data_list.append({"State": state_name, 
			"GeoID": geo_id, "lat": lat, "long": lon, "population": pop,
			"boundaries": boundaries})

	return data_list

def get_state_polygon(state_name, state_filename="cb_2018_us_state_500k.zip"):
	sf = shapefile.Reader(state_filename)

	geo_dict = sf.shapeRecords().__geo_interface__

	if geo_dict["type"] != "FeatureCollection":
		raise ValueError("Shapefile is not a feature collection")

	for record_and_geometry in geo_dict["features"]:
		name = record_and_geometry["properties"]["NAME"]

		if name.lower() != state_name.lower():
			continue

		boundaries_longlat = record_and_geometry["geometry"]["coordinates"]
		boundaries = []

		for boundary_longlat in flatten_boundary_list(boundaries_longlat):
			boundaries.append(np.array([[lat, lon] for lon,lat in boundary_longlat]))

		return boundaries
