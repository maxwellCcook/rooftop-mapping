
"""
Textural and Segmentation Feature Extraction

This script identifies which set of textural features is most relevant

Maxwell Cook, maxwell.cook@colorado.edu
"""

import os, glob
import rioxarray as rxr
import geopandas as gpd
import xarray as xr
import numpy as np

# Coordinate Ref. System
proj = 32618  # UTM Zone 18N

maindir = '/Users/max/Library/CloudStorage/OneDrive-Personal/mcook/earth-lab/opp-urban-fuels/rooftop-materials/'

# Area of interest
aoi_path = 'data/spatial/raw/dc_data/boundaries/district_of_columbia.gpkg'
aoi = gpd.read_file(aoi_path).to_crs(crs=proj)
envelope = aoi.envelope  # county boundary envelope

# Load the tiff files
tifs = glob.glob(os.path.join(maindir,'data/spatial/mod/dc_data/planet-data/texture/', '**', '*.tif'), recursive=True)

# Open the tif files and create a list of feature names
names = []  # to store the new band names
arrays = []  # to store the renamed arrays
for tif in tifs:
    print(os.path.basename(tif)[:-4])
    # Retrieve the band naming convention
    parts = os.path.basename(tif)[:-4].split('_')
    feature = parts[-2]
    window = parts[-1][0]

    print(f"Renaming band to {feature+window}")
    names.append(str(feature+window))

    # Open the image file
    img = rxr.open_rasterio(tif, masked=True, cache=False)
    img = img.rio.clip(aoi.geometry)
    arrays.append(img)

# Merge the arrays into a multi-band image stack
# Get the resolution of the first raster as a reference
print(arrays[0].rio.crs)
print(arrays[0].rio.resolution())
height, width = arrays[0].rio.resolution()[0], arrays[0].rio.resolution()[1]

# Merge the rasters
print("Merging arrays ...")

band_coords = range(1, len(arrays) + 1)

stack = xr.concat(arrays, dim=xr.DataArray(band_coords, dims='band', name='band'))

# Assign the CRS and band names
stack.rio.write_crs(f"EPSG:{proj}", inplace=True)
stack.attrs['long_name'] = names

print(
    f"Shape: {stack.shape}; \n"
    f"CRS:  {stack.rio.crs}; \n"
    f"NoData Value: {stack.rio.nodata}; \n"
    f"Bands: {stack.band}; \n"
    f"Band Names: {stack.long_name}; \n"
    f"Data Type: {stack[0].dtype}")

# Write the CRS and export the image stack
out_path = 'data/spatial/mod/dc_data/planet-data/dc_data_pssceneMNF_texture.tif'
stack.rio.to_raster(out_path, compress='zstd', zstd_level=9,
                    dtype='float32', driver='GTiff')  # export to GeoTIFF


#####################################
# Merge with the other raster stack #
#####################################

stack_path = 'data/spatial/mod/dc_data/planet-data/dc_data_psscene15b.tif'
stack_og = rxr.open_rasterio(stack_path, masked=True, cache=False)

final_stack = xr.concat([stack_og,stack], dim="band")

# Manage the band order and names
new_bands = ['coastal_blue', 'blue', 'green_i', 'green', 'yellow', 'red', 'rededge', 'nir',
             'ndre', 'vgnir', 'vrnir', 'ndbibg', 'ndbirg', 'mnf1', 'mnf2']
for band in names:
    new_bands.append(band)
print(new_bands)

# Assign the new band names and coords
final_stack.attrs['long_name'] = new_bands
total_bands = final_stack.band.size  # Total number of bands in the merged stack
new_band_coords = np.arange(1, total_bands + 1)  # Create a new array of band numbers
# Assign the new band numbers to the 'band' coordinate
final_stack = final_stack.assign_coords(band=new_band_coords)

print(
    f"Shape: {final_stack.shape}; \n"
    f"CRS:  {final_stack.rio.crs}; \n"
    f"NoData Value: {final_stack.rio.nodata}; \n"
    f"Bands: {final_stack.band}; \n"
    f"Band Names: {final_stack.long_name}; \n"
    f"Data Type: {final_stack[0].dtype}")

# Export the final image stack
out_path = 'data/spatial/mod/dc_data/planet-data/dc_data_psscene27b.tif'
final_stack.rio.to_raster(out_path, compress='zstd', zstd_level=9,
                          dtype='float32', driver='GTiff')  # export to GeoTIFF

