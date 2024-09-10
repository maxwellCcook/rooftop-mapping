"""
Supporting functions for the OPP rooftop materials mapping project

List of functions:
"""

import os
import numpy as np
import pandas as pd
import rioxarray as rxr
import pysptools.noise as noise
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.neighbors import KDTree

from functools import reduce
from rasterstats import zonal_stats
from osgeo import osr

import warnings
warnings.filterwarnings("ignore", message="'GeoDataFrame.swapaxes' is deprecated")
warnings.filterwarnings("ignore", message="Setting nodata to -999; specify nodata explicitly", category=UserWarning)


def print_raster(raster, open_file):
    """
    :param raster: input raster file
    :param open_file: should the file be opened or not
    :return: print statement with raster information
    """
    if open_file is True:
        img = rxr.open_rasterio(raster, masked=True).squeeze()
    else:
        img = raster
    print(
        f"shape: {img.rio.shape}\n"
        f"resolution: {img.rio.resolution()}\n"
        f"bounds: {img.rio.bounds()}\n"
        f"sum: {img.sum().item()}\n"
        f"CRS: {img.rio.crs}\n"
        f"NoData: {img.rio.nodata}"
    )
    del img


def band_correlations(da_in, out_png):
    """ Returns a correlation plot """
    # Convert to a numpy array and test band correlations
    image_np = da_in.values
    # Reshape the data to (bands, pixels)
    bands, height, width = image_np.shape
    image_np_t = image_np.reshape(bands, -1).T  # Transpose to get shape (pixels, bands)
    # Create a mask for non-NaN values
    valid_mask = ~np.isnan(image_np_t).any(axis=1)
    # Handle NaN for the correlation matrix
    image_np_tm = image_np_t[valid_mask]
    # Calculate the correlation matrix
    cor_mat_tm = np.corrcoef(image_np_tm, rowvar=False)

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cor_mat_tm, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Band Correlation Matrix')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show()

    # Clean up
    del image_np, image_np_t, image_np_tm, valid_mask, cor_mat_tm


def balance_sampling(df, ratio=5, strategy='undersample'):
    """
    Generate balanced sample from training data based on the defined ratio.
    This can be done with majority undersampling or minority oversampling ('strategy' parameter)
    Args:
        - df: the dataframe with rows as training data
        - ratio: the sampling ration (i.e., 5:1 for minority classes default)
    Returns:
        - random sample with class ratios as defined
    """

    # Get the class counts
    class_counts = df['class_code'].value_counts()
    min_class_count = class_counts.min()

    # Calculate the target count for each class based on the ratio
    target_count = {
        class_label: max(min_class_count, min(min_class_count * ratio, len(df[df['class_code'] == class_label])))
        for class_label in class_counts.index
    }

    # Create an empty list to store balanced dataframes
    balanced_dfs = []
    for class_label in class_counts.index:
        class_df = df[df['class_code'] == class_label]
        if strategy == 'undersample':
            # Under-sample the majority class, but do not undersample below its original count
            balanced_class_df = resample(
                class_df, replace=False, n_samples=target_count[class_label], random_state=42)
        elif strategy == 'oversample':
            # Over-sample the minority class
            balanced_class_df = resample(
                class_df, replace=True, n_samples=target_count[class_label], random_state=42)
        balanced_dfs.append(balanced_class_df)

    # Concatenate the results by class
    balanced_df = pd.concat(balanced_dfs)
    return balanced_df


def split_training_data(gdf, ts, vs):
    """
    Splits dataframe into train, test, and validation samples with the defined ratios
    Args:
        - gdf: training samples (geo data frame)
        - ts: test size #
        - vs: validation size #
    Returns:
        train, test, and validation dataframes
    """

    train_df, test_df, val_df = [], [], []

    for cl in gdf.class_code.unique():
        # subset to class
        _gdf = gdf.loc[gdf.class_code == cl]

        # get train and test validation arrays.
        # test array is validation array split in half.
        _train, _valtest = train_test_split(_gdf, random_state=27, test_size=ts)
        train_df.append(_train)

        _val, _test = train_test_split(_valtest, random_state=27, test_size=vs)
        test_df.append(_test)
        val_df.append(_val)

    # Concatenate the samples across classes
    all_train_df = pd.concat(train_df)
    all_train_df = gpd.GeoDataFrame(all_train_df, crs=gdf.crs)

    all_val_df = pd.concat(val_df)
    all_val_df = gpd.GeoDataFrame(all_val_df, crs=gdf.crs)

    all_test_df = pd.concat(test_df)
    all_test_df = gpd.GeoDataFrame(all_test_df, crs=gdf.crs)

    return all_train_df, all_val_df, all_test_df


def min_dist_sample(gdf, min_distance):
    """
    Filters the GeoDataFrame to ensure samples are at least min_distance apart.

    Args:
        gdf: GeoDataFrame containing 'geometry' column.
        min_distance: Minimum distance between samples in the same units as the geometry.

    Returns:
        Filtered GeoDataFrame.
    """
    coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])
    tree = KDTree(coords)
    indices_to_keep = set(range(len(gdf)))

    for i in range(len(gdf)):
        if i not in indices_to_keep:
            continue
        indices = tree.query_radius([coords[i]], r=min_distance)[0]
        for index in indices:
            if index != i:
                indices_to_keep.discard(index)

    del coords, tree, indices

    return gdf.iloc[list(indices_to_keep)]


class BandStatistics:
    """
    Class to handle sampling multi-band imagery at geometries in multiprocessing
    """
    def __init__(self, geom_path, raster_path, unique_id):
        """
        Initializes the BandStatistics object.
        Args:
            geom_path: Path to the input geospatial containing geometries
            raster_path: Path to the multi-band raster stack
        """
        # Load the geometries
        self.geometries = gpd.read_file(geom_path)  # Load the geometries
        # Load the raster data array
        self.raster = rxr.open_rasterio(raster_path)  # Use Dask chunking for efficient loading
        self.bands = self.raster.shape[0]  # Get the number of bands from the shape of the data
        self.nodataval = self.raster.rio.nodata  # Handle NoData values
        self.band_desc = list(self.raster.long_name)  # Get band descriptions or indices from the dataset
        self.band_num = list(self.raster.band.values)
        print(f"Raster contains {self.bands} bands: {self.band_desc}")

        self.id_col = str(unique_id)  # the unique identifier for geometries

    def compute_band_stats(self, geom_da, band_da, band, stat):
        """
        Computes band statistics for a single geometry or point for a specific band.

        Args:
            geom_da: geometry chunks for processing
            band_da: The raster band numpy array
            band: ...
            stat: statistic to be used (e.g., 'mean', 'median')
        """
        affine = self.raster.rio.transform()

        if geom_da.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).all():
            # print(f'Processing polygon geometries for band {self.band_desc[band - 1]}.')
            stats = zonal_stats(
                geom_da[[self.id_col, 'geometry']],
                band_da,
                affine=affine,
                stats=[stat],
                all_touched=True,
                nodata=self.nodataval,
                geojson_out=True  # Retain original attributes in the output
            )
            # Convert the list of dicts to a DataFrame, extract properties
            stats_df = pd.DataFrame(stats)
            # Tidy the columns
            stats_df[self.id_col] = stats_df['properties'].apply(lambda x: x.get(self.id_col))
            stats_df[stat] = stats_df['properties'].apply(lambda x: x.get(stat))
            stats_df = stats_df[[self.id_col, stat]]
            # Rename the band statistic column
            stats_df.rename(columns={stat: f'{self.band_desc[band - 1]}'}, inplace=True)

            return stats_df

        else:
            # print(f'Processing point geometries for band {self.band_desc[band - 1]}.')
            coord_list = [(x, y) for x, y in zip(geom_da.geometry.x, geom_da.geometry.y)]
            points_values = [band_da.rio.sample([coord]) for coord in coord_list]
            stats_df = geom_da.copy()
            stats_df[self.band_desc[band - 1]] = points_values

            return stats_df

    def process_chunk(self, chunk, stat):
        """
        Processes a chunk of geometries for band statistics across all bands.

        Args:
            chunk: A subset of the geometries to process
            stat: Which statistic to be used
        """
        results = None  # Initialize as None to handle merging

        for band in range(1, self.bands + 1):  # Iterate over each band
            band_data = self.raster.sel(band=band).values  # Convert to numpy array
            stats = self.compute_band_stats(chunk, band_data, band, stat)  # Process entire chunk at once

            # Merge the statistics for each band on the unique identifier (uid)
            if results is None:
                results = stats  # Initialize with the first band's stats
            else:
                results = results.merge(stats, on=self.id_col, how='left')  # Merge on uid

        return results

    def parallel_compute_stats(self, stat):
        """
        Parallelizes the band statistics computation for all geometries across all bands.
        Automatically sets the number of workers to the number of available CPU cores minus one.

        Args:
            stat: The statistic to compute (e.g., 'mean', 'median', etc.)
        """
        # Get the number of CPU cores and set workers to one less than available cores
        num_workers = max(1, os.cpu_count() - 1)
        print(f"Using {num_workers} workers.")

        # Split geometries into chunks for parallel processing
        chunks = np.array_split(self.geometries, num_workers)

        # Parallel processing using multiprocessing Pool
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(self.process_chunk, [(chunk, stat) for chunk in chunks])

        # Flatten the list of results and convert to DataFrame
        results_df = pd.concat(results, ignore_index=True)

        return results_df


def get_coords(frame):
    xy = frame.geometry.xy
    x = xy[0].tolist()
    y = xy[1].tolist()
    return [list(z) for z in zip(x, y)]


def array_to_tif(arr, ref, out_path, dtype, clip=False, shp=None):
    # Save the MNF transformed data as a raster
    # Transpose the new array before exporting
    in_arr = arr.transpose(2, 1, 0)
    print(in_arr.shape)
    # Assign the correct coordinates for the transposed 'y' dimension
    band_coords = range(in_arr.shape[0])
    y_coords = ref.y.values
    x_coords = ref.x.values
    # Store the new array and export
    out_arr = xr.DataArray(
        in_arr,
        dims=("band", "y", "x"),
        coords={
            "band": band_coords,
            "y": y_coords,
            "x": x_coords,
        }
    )
    # Export the new DataArray as a new GeoTIFF file
    out_arr.rio.set_crs(ref.rio.crs)  # Set the CRS
    out_arr.rio.write_transform(ref.rio.transform())  # Set the GeoTransform
    if clip is True and shp is not None:
        print("Clipping raster array ...")
        out_arr = out_arr.rio.clip(shp.geometry)
    out_arr.rio.to_raster(out_path, compress='zstd', zstd_level=9,
                          dtype=dtype, driver='GTiff')  # export to GeoTIFF

    print(f"Successfully exported array to '{out_path}'")

    return out_arr


def pixel_to_xy(pixel_pairs, gt=None, wkt=None, path=None, dd=False):
    """
    Modified from code by Zachary Bears (zacharybears.com/using-python-to-
    translate-latlon-locations-to-pixels-on-a-geotiff/).
    This method translates given pixel locations into longitude-latitude
    locations on a given GeoTIFF. Arguments:
        pixel_pairs The pixel pairings to be translated in the form
                    [[x1, y1],[x2, y2]]
        gt          [Optional] A GDAL GeoTransform tuple
        wkt         [Optional] Projection information as Well-Known Text
        path        The file location of the GeoTIFF
        dd          True to use decimal degrees for longitude-latitude (False
                    is the default)

    NOTE: This method does not take into account pixel size and assumes a
            high enough image resolution for pixel size to be insignificant.
    """

    pixel_pairs = map(list, pixel_pairs)
    srs = osr.SpatialReference()  # Create a spatial ref. for dataset
    srs.ImportFromWkt(wkt)

    # Go through all the point pairs and translate them to pixel pairings
    xy_pairs = []
    for point in pixel_pairs:
        # Translate the pixel pairs into untranslated points
        lon = point[0] * gt[1] + gt[0]
        lat = point[1] * gt[5] + gt[3]
        xy_pairs.append((lon, lat))  # Add the point to our return array

    return xy_pairs


def mnf_transform(data_arr, n_components=3, nodata=-9999):
    """
        Applies the MNF rotation to a raster array; returns in HSI form
        (m x n x p). Arguments:
            rast    A NumPy raster array
            nodata  The NoData value
    """
    arr = data_arr.copy().transpose()
    arr[arr == nodata] = 0  # Remap any lingering NoData values

    # Apply the Minimum Noise Fraction (MNF) rotation
    mnf = noise.MNF()
    mnf_arr = mnf.apply(arr)
    if n_components is not None:
        return mnf_arr  # return the entire array
    else:
        print(f"Returning {n_components} components ...")
        return mnf_arr.get_components(n_components)  # return n components


def ravel_and_filter(arr, cleanup=True, nodata=-9999):
    """
    Reshapes a (p, m, n) array to ((m*n), p) where p is the number of
    dimensions and, optionally, filters out the NoData values. Assumes the
    first axis is the shortest. Arguments:
        arr      A NumPy array with shape (p, m, n)
        cleanup  True to filter out NoData values (otherwise, only ravels)
        nodata   The NoData value; only used in filtering
    """
    shp = arr.shape
    # If the array has already been raveled
    if len(shp) == 1 and cleanup:
        return arr[arr != nodata]
    # If a "single-band" image
    if len(shp) == 2:
        arr = arr.reshape(1, shp[-2] * shp[-1]).swapaxes(0, 1)
        if cleanup:
            return arr[arr != nodata]
    # For multi-band images
    else:
        arr = arr.reshape(shp[0], shp[1] * shp[2]).swapaxes(0, 1)
        if cleanup:
            return arr[arr[:, 0] != nodata]
    return arr


def get_spectra(cube, coord_list, gt, wkt):
    """
    Returns the spectral profile of the pixels indicated by the indices
    provided. NOTE: Assumes an HSI cube (transpose of a GDAL raster).
    Arguments:
        hsi_cube    An HSI cube (n x m x p)
        idx         An array of indices that specify one or more pixels in a
                    raster
                    :param wkt:
                    :param gt:
                    :param coord_list:
                    :param cube:
    """

    def spectra_at_idx(data_arr, idx):
        return np.array([data_arr[p[0], p[1], :] for p in idx])

    xy_pairs = map(list, coord_list)
    srs = osr.SpatialReference()  # Create a spatial ref. for dataset
    srs.ImportFromWkt(wkt)  # Set up coord. transform.
    # Go through all the point pairs and translate them to lng-lat pairs
    pixel_pairs = []
    for point in xy_pairs:
        # Translate the x and y coordinates into pixel values
        x = (point[0] - gt[0]) / gt[1]
        y = (point[1] - gt[3]) / gt[5]
        pixel_pairs.append((int(x), int(y)))  # Add point to our return array
    # Get the pixel values at coordinates
    return spectra_at_idx(data_arr=cube.transpose(), idx=pixel_pairs)


def convex_hull_graham(points, indices=False):
    """
    Returns points on convex hull of an array of points in CCW order according
    to Graham's scan algorithm. By Tom Switzer <thomas.switzer@gmail.com>.
    Arguments:
        points      The points for which a convex hull is sought
        indices     True to return a tuple of (indices, hull)
    """
    TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

    def cmp(a, b):
        return (a > b) - (a < b)

    def turn(p, q, r):
        return cmp((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]), 0)

    def keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
        if not len(hull) or hull[-1] != r:
            hull.append(r)
        return hull

    pts_sorted = sorted(points)
    l = reduce(keep_left, pts_sorted, [])
    u = reduce(keep_left, reversed(pts_sorted), [])
    hull = l.extend(u[i] for i in range(1, len(u) - 1)) or l

    if indices:
        return [points.index(h) for h in hull], hull

    return hull


def lsma(cases, members):
    # For regular LSMA with single endmember spectra
    am = amap.FCLS()
    # c is number of pixels, k is number of bands
    cc, kk = cases.shape if len(cases.shape) > 1 else (1, cases.shape[0])
    print(f"c: {cc}; k: {kk}")
    return am.map(cases.reshape((1, cc, kk)), members, normalize=False)


def mesma(array_pairs):
    am = amap.FCLS()
    # For multiple endmember spectra, in chunks
    cases, endmembers = array_pairs
    # c is number of pixels, k is number of bands
    c, k = cases.shape if len(cases.shape) > 1 else (1, cases.shape[0])
    return [
        am.map(cases[i, ...].reshape((1, 1, k)), endmembers[i, ...], normalize=False) for i in range(0, c)
    ]


def partition(array, processes, axis=0):
    """
    Creates index ranges for partitioning an array to work on over multiple
    processes. Arguments:
        array           The 2-dimensional array to partition
        processes       The number of processes desired
    """
    N = array.shape[axis]
    P = (processes + 1)  # Number of breaks (number of partitions + 1)
    # Break up the indices into (roughly) equal parts
    partitions = list(zip(np.linspace(0, N, P, dtype=int)[:-1],
                          np.linspace(0, N, P, dtype=int)[1:]))
    # Final range of indices should end +1 past last index for completeness
    work_indices = partitions[:-1]
    work_indices.append((partitions[-1][0], partitions[-1][1] + 1))
    return work_indices

