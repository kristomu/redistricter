'''
Parses TABBLOCK20 census block databases, outputting a list of dicts where
each dict contains the state name, Geo ID, internal latitude,
internal longitude, and population count.
'''

''' The following might be of interest later, if I'm going to divide
census blocks linearly:

import shapefile
sf = shapefile.Reader("tl_2023_01_tabblock20.zip")
shapes = sf.shapes()
shapes[0].points

Then I'd define this shape on a sphere (ignoring altitude) and use a
linear/uniform interpolation: if the current detail level gives a block
x points, then these are each counted as having an xth fraction of the
block's total population. All that's required is to efficiently do
a point in polygon test.

'''

import csv
import tqdm
import os, glob
from dbfread import DBF

def get_state_names(state_file="national_state2020.txt"):
	state_csv = csv.DictReader(
		open(state_file, "r"), delimiter="|")

	state_names = {}

	for state_row in state_csv:
		state_names[int(state_row["STATEFP"])] = (state_row["STATE"], state_row["STATE_NAME"])
		state_names[state_row["STATE"]] = state_row["STATE_NAME"]
		state_names[state_row["STATE_NAME"]] = state_row["STATE"]

	return state_names

def get_census_block_data(dbf_file, state_names):
	data_list = []

	for record in DBF(dbf_file):
		state_name = state_names[int(record["STATEFP20"])][1]
		geo_id = record["GEOID20"]
		lat = float(record["INTPTLAT20"])
		lon = float(record["INTPTLON20"])
		pop = int(record["POP20"])
		data_list.append({"State": state_name, 
			"GeoID": geo_id, "lat": lat, "long": lon, "population": pop})

	return data_list

def get_all_data(state_names):
	data_list = []
	dbf_files = list(glob.glob("*.dbf"))

	for dbf_file in tqdm.tqdm(dbf_files):
		data_list += get_census_block_data(dbf_file, state_names)

	return data_list

def write_census_block_csv():
	state_names = get_state_names()
	block_data = get_all_data(state_names)

	columns = list(block_data[0].keys())

	with open("census_blocks.csv", "w") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=columns)
		writer.writeheader()

		for row in block_data:
			writer.writerow(row)

def write_per_state_census_block_csvs():
	state_names = get_state_names()

	dbf_files = list(glob.glob("*.dbf"))

	for dbf_file in tqdm.tqdm(dbf_files):
		block_data = get_census_block_data(dbf_file, state_names)
		columns = list(block_data[0].keys())

		state_shorthand = state_names[block_data[0]["State"]]

		csv_file = open("census_blocks_" + state_shorthand + ".csv", "w")
		writer = csv.DictWriter(csv_file, fieldnames=columns)
		writer.writeheader()

		for row in block_data:
			writer.writerow(row)