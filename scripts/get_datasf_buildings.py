"""Download the official San Francisco building footprints from DataSF and crop them
to the demo area.  Run from the repository root:

    python scripts/get_datasf_buildings.py

Needs:  pip install pandas geopandas shapely requests
Source: DataSF "Building Footprints" (dataset id ynuv-fyni), public domain licence (PDDL).

If the download is slow or blocked, download the CSV by hand from
https://data.sfgov.org/d/ynuv-fyni  (Export > CSV) and save it as
data/raw/sf_building_footprints.csv, then run this script again: it skips the download.
"""
import os
import sys

import geopandas as gpd
import pandas as pd
import requests
from shapely import wkt

URL = "https://data.sfgov.org/api/views/ynuv-fyni/rows.csv?accessType=DOWNLOAD"
RAW = "data/raw/sf_building_footprints.csv"      # big file: stays out of Git (see data/raw/.gitignore)
OUT = "data/raw/sf_buildings_datasf.geojson"     # cropped file: this one is committed
CENTER_LAT, CENTER_LON = 37.7600, -122.4300      # same centre as get_sf_data.py
RADIUS_M = 3000

os.makedirs("data/raw", exist_ok=True)

if not os.path.exists(RAW):
    print("Downloading DataSF building footprints (large file, can take a few minutes)...")
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(RAW, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

df = pd.read_csv(RAW, low_memory=False)
print(len(df), "footprints in the full dataset")
print("Columns:", list(df.columns))

# Find the column that holds the polygon geometry (text starting with POLYGON / MULTIPOLYGON)
geom_col = None
for c in df.columns:
    s = df[c].dropna().astype(str)
    if len(s) and s.iloc[0].strip().upper().startswith(("POLYGON", "MULTIPOLYGON")):
        geom_col = c
        break
if geom_col is None:
    sys.exit("No polygon column found. Open the CSV, find the geometry column, and tell the team.")
print("Geometry column:", geom_col)

df = df.dropna(subset=[geom_col])
gdf = gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=df[geom_col].apply(wkt.loads), crs="EPSG:4326")

# Crop to RADIUS_M around the demo centre (work in metres using UTM zone 10N)
centre = gpd.GeoSeries(gpd.points_from_xy([CENTER_LON], [CENTER_LAT]), crs="EPSG:4326").to_crs(epsg=32610).iloc[0]
gdf_m = gdf.to_crs(epsg=32610)
near = gdf_m[gdf_m.geometry.centroid.distance(centre) <= RADIUS_M].to_crs(epsg=4326)

near.to_file(OUT, driver="GeoJSON")
print("Saved", len(near), "footprints to", OUT)
print("Tip: DataSF adds height and other columns that OpenStreetMap lacks. Check the column list above.")
