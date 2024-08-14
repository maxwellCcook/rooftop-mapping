
"""
Dimensionality reduction of multi band raster array
Minimum Noise Fraction (MNF) transformation
w/ 'PySpTools' denoising
"""

import os, sys
import geopandas as gpd
import rioxarray as rxr
import earthpy.plot as ep
from matplotlib import pyplot as plt
import pysptools.util as sp_utils

# Custom functions
sys.path.append(os.path.join(os.getcwd(),'code/'))
from __functions import *

# Coordinate Ref. System
proj = 32618  # UTM Zone 18N

# Area of interest
aoi_path = 'data/spatial/raw/dc_data/boundaries/district_of_columbia.gpkg'
aoi = gpd.read_file(aoi_path).to_crs(crs=proj)
envelope = aoi.envelope  # county boundary envelope

# Bring in the image file and get the projection information
image_path = 'data/spatial/mod/dc_data/planet-data/dc_data_psscene13b.tif'
stack = rxr.open_rasterio(image_path, masked=True, cache=False, tiled=True)
stack - stack.fillna(np.nan)
stack_ = stack[:8, :, :]  # Keep the original 8-bands for the MNF
print(
    f"Shape: {stack_.shape}; \n"
    f"CRS:  {stack_.rio.crs}; \n"
    f"NoData Value: {stack_.rio.nodata}; \n"
    f"Bands: {stack_.band}; \n"
    f"Band Names: {stack_.long_name}; \n"
    f"Data Type: {stack_[0].dtype}")


##################################
# Dimensionality reduction (PCA) #
##################################

# Convert nodata to -9999 for the MNF to work
stack_ = stack_.fillna(-9999)  # fill NA value for the MNF processing
stack_arr = stack_.to_numpy()  # convert to numpy ndarray
print(f"Image array shape: {stack_arr.shape}; and type: {type(stack_arr)}")

# Perform the MNF on the array
stack_mnf = mnf_transform(stack_arr, n_components=8)  # Transform the "cube" using # components = bands
print(
    f"MNF image shape: {stack_mnf.shape}; \n"
    f"transposed shape: {stack_mnf.T.shape}; \n"
    f"and type: {type(stack_mnf)};")

# Explore the dimensionality of the data
# Filter out NoData values from the MNF-transformed image
filtered = ravel_and_filter(np.where(stack_arr == -9999, -9999, stack_mnf.T))
# Obtain the covariance matrix
cov_m = sp_utils.cov(filtered)
# Compute the eigenvalues, sort them, reverse the sorting
eigenvals = np.sort(np.linalg.eig(cov_m)[0])[::-1]
eigenvals_p = np.power(eigenvals, 2) / sum(np.power(eigenvals, 2))
print(f"Variance explained by MNF rotations: \n {list(map(lambda x: round(x, 5), eigenvals_p.tolist()))}")

# Plot the variance explained
eigenvals_p_ = list(map(lambda x: round(x, 3), eigenvals_p.tolist()))
plt.figure(figsize=(10, 5))
components = range(1, 1 + len(eigenvals_p_))
plt.plot(components, eigenvals_p_, marker='o', linestyle='-', color='b', label='Variance explained')

# Add labels for the first two components
for i, txt in enumerate(eigenvals_p_[:2]):
    plt.annotate(f"{txt}", (components[i], eigenvals_p_[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.ylabel('Proportion of Variance Explained')
plt.xlabel('MNF Component')
plt.xticks(components)  # Ensure x-ticks match the number of components
plt.title('Proportion of Variance Explained by MNF Components')
plt.legend(loc='best')
plt.show()


# Clear up some memory
del filtered, cov_m, eigenvals, eigenvals_p, eigenvals_p_, stack_


###########################################################
# MNF1 explains >98% of the variance in the original bands

# Export the MNF1 array to a GeoTIFF
out_path = 'data/spatial/mod/dc_data/planet-data/dc_data_psscene8b_mnf.tif'
rast = array_to_tif(stack_mnf[:,:,:2], stack, out_path, 'float32', clip=True, shp=aoi)  # only keep the first two

# Combine the original bands and MNF transformations

# Create a new coordinate for the 'band' dimension that includes these new bands
original_bands = stack.band.values.tolist()
all_band_names = original_bands + ['mnf1', 'mnf2']
# Convert to DataArray
mnf_array = xr.DataArray(data=stack_mnf[:, :, :2].T,
                         dims=['band','y', 'x'],
                         coords={'y': stack.coords['y'], 'x': stack.coords['x'], 'band': all_band_names[-2:]},
                         attrs=stack.attrs)
# Clip to the AOI
mnf_array.rio.write_crs(proj, inplace=True)
mnf_array = mnf_array.rio.clip(aoi.geometry)

# Concatenate all bands
stack_out = xr.concat([stack, mnf_array], dim='band')
# Update 'long_name' attribute
new_bands = ['coastal_blue', 'blue', 'green_i', 'green', 'yellow', 'red', 'rededge', 'nir',
             'ndre', 'vgnir', 'vrnir', 'ndbibg', 'ndbirg', 'mnf1', 'mnf2']
stack_out.attrs['long_name'] = new_bands
print(
    f"Shape: {stack_out.shape}; \n"
    f"CRS:  {stack_out.rio.crs}; \n"
    f"NoData Value: {stack_out.rio.nodata}; \n"
    f"Bands: {stack_out.band}; \n"
    f"Band Names: {stack_out.long_name}; \n"
    f"Data Type: {stack_out[0].dtype}")

# Plot the final bands (MNF + Indices)
ep.plot_bands(stack_out, figsize=(10,8))

print(f"Number of bands: {len(new_bands)}")
out_path = 'data/spatial/mod/dc_data/planet-data/dc_data_psscene15b.tif'
stack_out.rio.to_raster(out_path, compress='zstd', zstd_level=9,
                        dtype='float32', driver='GTiff')  # export to GeoTIFF
