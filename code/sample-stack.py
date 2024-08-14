
"""
Convert building footprints to centroid and sample the image stack
"""

# Packages
import os, time, sys
import geopandas as gpd
import pandas as pd
import rasterio as rio
from concurrent.futures import ProcessPoolExecutor

# Custom functions
sys.path.append(os.path.join(os.getcwd(),'code/'))
from __functions import *

print(os.getcwd())

# Coordinate Ref. System
proj = 32618  # UTM Zone 18N

maindir = '/Users/max/Library/CloudStorage/OneDrive-Personal/mcook/earth-lab/opp-urban-fuels/rooftop-materials/'

begin = time.time()  # start time


#############
# Functions #
#############

# Function to sample raster values to points for multi-band image
def img_vals_at_pts(img, points, band_names):
    coord_list = [(x, y) for x, y in zip(points["geometry"].x, points["geometry"].y)]
    n_bands = img.count
    print(f"Number of bands to process: {n_bands}")
    for i in range(0,n_bands):
        band = desc[i]
        print(str(i)+'_'+band)
        points[f"{band}"] = [x for x in img.sample(coord_list, indexes=i+1)]
    points_df = points.reset_index()
    points_df[desc] = points_df[band_names].astype(np.float32)
    return points_df


# Zonal statistics function for image data and polygons

def img_vals_in_poly(img_path, polys, band_names, nodataval, stat='mean'):
    # Create a copy of the polygons to store the results
    stats_df = polys.copy()

    # Calculate the number of cores to use, reserving 2 cores
    num_cores = os.cpu_count()
    if num_cores is not None:  # os.cpu_count() can return None
        max_workers = max(1, num_cores - 1)  # Reserve 2 cores, but ensure at least 1 worker
    else:
        max_workers = 1  # Default to 1 worker if os.cpu_count() is None

    # Set up parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for band, band_name in enumerate(band_names, start=1):
            futures.append(executor.submit(compute_band_stats, band, img_path, polys, stat, nodataval))

        for future in futures:
            result = future.result()
            band = list(result.keys())[0]
            stats_df[f'band_{band}'] = result[band]

    # Optionally, rename columns based on band names
    band_name_mapping = {f'band_{i + 1}': name for i, name in enumerate(band_names)}
    stats_df.rename(columns=band_name_mapping, inplace=True)

    return stats_df


#############
# Data Prep #
#############

# Load the footprint data
filename = 'data/spatial/raw/dc_data/boundaries/dc_data_ocm_w_ztrax_matched.gpkg'
dc_data = gpd.read_file(filename)

# Retain needed columns and tidy the data
# Convert to centroids

gdf = dc_data.to_crs(proj)
# Create the area attribute
gdf['areaUTM'] = [geom.area for geom in gdf.geometry]
gdf['areaUTMsqft'] = [geom.area*10.7639 for geom in gdf.geometry]
# filter the footprints by distance metric and area mismatch
gdf = gdf.loc[(gdf._distance <= 10) & (gdf.areaUTMsqft <= gdf.LotSizeSquareFeet)]

# add the class code categorical variable
gdf['class_code'] = gdf.RoofCoverStndCode.astype('category')  # category type is required for encoding
print(gdf['class_code'].describe(include='all'))

# Filter out footprints below the 10th percentile of size for that class
gdfs = []
for cls in gdf['class_code'].unique():
    print(cls)
    # Filter to that class
    gdf_cls = gdf[gdf['class_code'] == cls]
    # Calculate the 10th percentile in building size
    p10 = np.percentile(gdf_cls['areaUTMsqft'], 10)
    gdf_cls = gdf_cls[gdf_cls['areaUTMsqft'] > p10]  # filter based on the 10th percentile
    # append to the output list
    gdfs.append(gdf_cls)

    del p10, gdf_cls

# Merge them back
gdf = pd.concat(gdfs, ignore_index=True)

# Retain required columns
gdf = gdf[['class_code','areaUTMsqft','geometry']].reset_index(drop=True)
# Create a unique ID column
gdf['uid'] = gdf.index + 1
gdf['uid'] = gdf['uid'].astype(str) + gdf['class_code'].astype(str)

# Join to the description as well
lookup = pd.read_csv(os.path.join(maindir,'data/tabular/raw/variable_lookup/RoofCoverStndCode_encoding.csv'))
lookup = lookup[['Description','Code']]
lookup = lookup.rename(columns={"Code": "class_code","Description": "description"})
gdf = gdf.merge(lookup, on='class_code')

del lookup

print(gdf.head())


# Create the train/test points
print("... Tidying the GeoData and filtering bad classes ...")

# Handle 'bad' classes
bad_classes = ['','BU','OT']

out_gdfs = []
for cl in gdf.class_code.unique():
    print(cl)

    if cl in bad_classes:
        print(f'Skipping {cl} class')
        continue

    _gdf = gdf.loc[gdf.class_code == cl]

    # skip small sample size
    if _gdf.shape[0] < 10:
        print(f'Class {cl} has shape {_gdf.shape} ... skipping ...')
        continue

    out_gdfs.append(_gdf)

# Concatenate the reference data
reference = pd.concat(out_gdfs)
print(reference['class_code'].unique())
print(reference['class_code'].value_counts())

# Save the cleaned footprint data
reference['class_code'] = reference['class_code'].astype(str)
reference.to_file(f'data/spatial/mod/dc_data/training/dc_data_reference_footprints.gpkg')

# Create the centroids for point sampling
reference_pt = reference.copy()
reference_pt['geometry'] = reference_pt['geometry'].centroid
reference_pt.to_file(f'data/spatial/mod/dc_data/training/dc_data_reference_centroids.gpkg')

# Tidy up the memory
del _gdf, bad_classes, gdf, gdfs, out_gdfs, dc_data


#########################
# Sample the image data #
#########################

# Load the image data (27-band: original bands, spectral indices, MNF1 & MNF2, textural features (3,5,7))
stack_path = 'data/spatial/mod/dc_data/planet-data/dc_data_psscene27b.tif'
mosaic = rio.open(stack_path)

# Grab some metadata
desc = list(mosaic.descriptions)
metadata = mosaic.meta
print(f'Raster description: {desc}; \n Metadata: {metadata}')
if isinstance(mosaic, rio.io.DatasetReader):
    print("The object is a rasterio dataset.")
# Also grab the NoData value
nodata = mosaic.nodata

# Sample all the centroids by roof material type
all_vals = img_vals_at_pts(mosaic,reference_pt,desc)
# Check on the results
print(all_vals.head())
print(all_vals['ndre'].describe)  # check one column
print(all_vals.columns.values.tolist())
# Write to a gpkg and csv
all_vals = all_vals.to_crs(proj)
all_vals.to_file('data/spatial/mod/dc_data/training/dc_data_reference_sampled.gpkg')
all_vals.drop('geometry',axis=1).to_csv('data/tabular/mod/dc_data/training/dc_data_reference_sampled.csv')
print(f"Sampling of the footprint centroids completed: {round((time.time() - begin)/60)} minutes")

del all_vals

# Now, sample the polygon data (footprint average)
# Run in parallel
if __name__ == "__main__":
    begin2 = time.time()
    all_vals_poly = img_vals_in_poly(stack_path, reference, desc, nodata)
    print(all_vals_poly.head())
    print(f"Sampling of the footprint polygons completed: Additional {round((time.time() - begin2)/60)} minutes")

    # Create a mapping from band number to long name
    band_name_mapping = {i+1: name for i, name in enumerate(desc)}
    # Rename the columns using the mapping
    all_vals_poly = all_vals_poly.rename(columns=band_name_mapping)

    all_vals_poly.to_csv('data/tabular/mod/dc_data/training/dc_data_reference_sampled_footprint.csv')

print("Complete!")
print(f"Total elapsed time: {round((time.time() - begin)/60)} minutes.")