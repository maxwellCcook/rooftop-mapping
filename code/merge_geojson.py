
import os, glob
import geopandas as gpd
import pandas as pd

proj = 32613

maindir = '/Users/max/Library/CloudStorage/OneDrive-Personal/mcook/earth-lab/opp-rooftop-mapping'

geojson_dir = os.path.join(maindir,'data/spatial/raw/denver_data/boundaries/footprints_geojson/')
print(geojson_dir)

geofiles = glob.glob(geojson_dir+'*.geojson')

gdfs = []
for geo in geofiles:
    print(f'Processing {os.path.basename(geo)}')
    gdf = gpd.read_file(geo).to_crs(proj)
    gdfs.append(gdf)

ref = pd.concat(gdfs)
print(ref.head())

ref.to_file(os.path.join(maindir,'data/spatial/raw/denver_data/training/denver_data_ocm_w_ztrax_matched_.gpkg'))