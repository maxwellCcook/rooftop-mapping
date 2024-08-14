
"""
Supporting functions for the OPP rooftop materials mapping project

List of functions:



"""

import numpy as np
import rioxarray as rxr
import pysptools.noise as noise
from osgeo import osr
from functools import reduce
import xarray as xr
from rasterstats import zonal_stats


def print_raster(raster,open_file):
    """
    :param raster: input raster file
    :param open_file: should the file be opened or not
    :return: print statement with raster information
    """
    if open_file is True:
        img = rxr.open_rasterio(raster,masked=True).squeeze()
    else:
        img = raster
    print(
        f"shape: {img.rio.shape}\n"
        f"resolution: {img.rio.resolution()}\n"
        f"bounds: {img.rio.bounds()}\n"
        f"sum: {img.sum().item()}\n"
        f"CRS: {img.rio.crs}\n"
        f"NoData: {img.rio.nodata}"
        f"Array: {img}"
    )
    del img


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
        xy_pairs.append((lon, lat)) # Add the point to our return array

    return xy_pairs


def mnf_transform(data_arr,n_components=3,nodata=-9999):
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
        arr = arr.reshape(1, shp[-2]*shp[-1]).swapaxes(0, 1)
        if cleanup:
            return arr[arr != nodata]
    # For multi-band images
    else:
        arr = arr.reshape(shp[0], shp[1]*shp[2]).swapaxes(0, 1)
        if cleanup:
            return arr[arr[:,0] != nodata]
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
        return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

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


# Zonal stats for parallel
def compute_band_stats(band, img_path, polys, stat, nodataval):
    stats = zonal_stats(
        polys.geometry,
        img_path,
        stats=[stat],
        band_num=band,
        all_touched=True,
        nodata=nodataval,
        geojson_out=True
    )
    return {band: [feature['properties'][stat] for feature in stats]}
