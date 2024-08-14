
"""
Pre-processing and correction for PlanetScope composites
Using multiple overlapping PlanetScope SuperDove scenes (8-band)

Calculate spectral indices
    - A
    - B
    - C

"""


import os, time
from glob import glob
import rioxarray as rxr
import xarray as xr
import numpy as np
from rioxarray.merge import merge_arrays
from rasterio.crs import CRS
import earthpy.plot as ep
import rasterio as rio
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')

begin = time.time()  # start time

# Load the environment variables

# Coordinate Ref. System
proj = 32618  # UTM Zone 18N (Washington, D.C.)

# PSScene directory
psscenes = glob('data/spatial/raw/dc_data/planet-data/PSScene8Band/*/*_SR_8b_clip.tif')

# Area of interest
aoi_path = 'data/spatial/raw/dc_data/boundaries/district_of_columbia.gpkg'
aoi = gpd.read_file(aoi_path).to_crs(crs=proj)
envelope = aoi.envelope  # county boundary envelope

# Process and create the mosaic
tiles = []
for i in range(len(psscenes)):
    print(os.path.basename(psscenes[i]))
    tile = rxr.open_rasterio(
        psscenes[i],masked=True,cache=False,chunks=True,lock=False
    ).squeeze().astype(rio.uint16)
    tiles.append(tile)  # append to the empty list
    del tile

# Get the resolution of the first raster as a reference
print(tiles[0].rio.crs)
print(tiles[0].rio.resolution())
height, width = tiles[0].rio.resolution()[0], tiles[0].rio.resolution()[1]

# Merge the rasters
print("Merging arrays ...")

mosaic = merge_arrays(
    dataarrays=tiles,
    res=(height, abs(width)),
    crs=CRS.from_epsg(proj),
    nodata=0,
    method='max'
).rio.clip(aoi.geometry)  # create the mosaic and clip to roi

time.sleep(1)
print(time.time() - begin)

# Write to disk
print("Writing mosaic image ...")

out_img = 'data/spatial/mod/dc_data/planet-data/dc_data_psscene8b.tif'
mosaic.rio.to_raster(
    out_img, compress='zstd', zstd_level=9,
    dtype='uint16', driver='GTiff')
tiles = []  # clear the list

print(f"Successfully exported the 8-band composite: {out_img}")

time.sleep(1)
print(round(time.time() - begin))

del mosaic, tiles, height, width   # clean up


##############################
# Calculate Spectral Indices #
##############################

print("Adding spectral indices to image composite ...")

begin = time.time()  # start time

stack = rxr.open_rasterio(out_img, masked=True, chache=False)

# image = image_data.where(image_data > 0, drop=True)  # drop 0 vals (NoData)
shp, gt, wkt, nd = stack.shape, stack.spatial_ref.GeoTransform, stack.rio.crs, stack.rio.nodata
print(
    f"Shape: {shp}; \n"
    f"GeoTransform: {gt}; \n"
    f"WKT: {wkt}; \n"
    f"NoData Value: {nd}; \n"
    f"Bands: {stack.band}; \n"
    f"Band Names: {stack.attrs['long_name']}; \n"
    f"Data Type: {stack[0].dtype}")

# copy attributes from the Green band (arbitrary, could be any band)
ndre = stack[3].copy(data=(stack[7] - stack[6]) / (stack[7] + stack[6]))
vgnir = stack[3].copy(data=(stack[3] - stack[7]) / (stack[3] + stack[7]))  # 'green','nir'
vrnir = stack[3].copy(data=(stack[5] - stack[7]) / (stack[5] + stack[7]))  # 'red','nir'
ndbibg = stack[3].copy(data=(stack[1] - stack[3]) / (stack[1] + stack[3]))  # 'blue','green'
ndbirg = stack[3].copy(data=(stack[5] - stack[3]) / (stack[5] + stack[3]))  # 'red','green'

# Put the indices into a list
si_arrays = [ndre,vgnir,vrnir,ndbibg,ndbirg]  # create a list of images

# Check a plot of the data, export
names = ['ndre','vgnir','vrnir','ndbibg','ndbirg']  # list of names for the bands
for i in range(len(si_arrays)):
    img = si_arrays[i]  # the band we are working with
    ep.plot_bands(img,scale=False,vmin=-1, vmax=1,title=names[i],figsize=(6,6))
    print(names[i])

# Create the final multi-band image stack
stack = xr.concat([stack, xr.concat(si_arrays, dim='band')], dim='band').astype(np.float32)

# Update the long_name attribute to include the other names
long_names = list(stack.attrs['long_name'])+names
stack.attrs['long_name'] = tuple(long_names)
print(f"shape: {stack.shape}; long_name: {stack.attrs['long_name']}")

# Test by plotting the 9th band (should be vgnir)
ep.plot_bands(
    stack[9].squeeze(),
    scale=False,
    vmin=-1,vmax=1,
    title="VGNIR",
    figsize=(6,6))

# Write the tiff file out
out_path = f'data/spatial/mod/dc_data/planet-data/dc_data_psscene13b.tif'
stack.rio.to_raster(out_path,compress='zstd', zstd_level=9, driver='GTiff', dtype='float32')

print("Successfully exported the 13-band stack w/ indices ...")

time.sleep(1)
print(round(time.time() - begin))

