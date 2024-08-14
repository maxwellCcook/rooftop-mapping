"""
Classify the PlanetScope Imagery based on the best performing scenario for each roof type
Generate the probability surface for each roof material type
Extract the zonal statistics for building footprints (the average probability for each class)

maxwell.cook@colorado.edu
"""

import os, time
import pandas as pd
import rioxarray as rxr
import xgboost as xgb
import numpy as np
import xarray as xr

maindir = '/Users/max/Library/CloudStorage/OneDrive-Personal/mcook/earth-lab/opp-urban-fuels/rooftop-materials'

begin = time.time()


# ------ Functions ------- #

def print_raster(raster):
    print(
        f"shape: {raster.rio.shape}\n"
        f"resolution: {raster.rio.resolution()}\n"
        f"bounds: {raster.rio.bounds()}\n"
        f"sum: {raster.sum().item()}\n"
        f"CRS: {raster.rio.crs}\n"
    )


def flatten_raster(raster):
    """Flatten a raster to a 2D array where each row is a pixel and each column is a band."""
    # Number of bands is the first dimension of the raster
    bands, height, width = raster.shape

    # Reshape the raster so that each row represents a pixel, and each column represents a band
    flattened_raster = raster.values.reshape(bands, height * width).T  # Transpose to get (height*width, bands)

    return flattened_raster


#############
# Load data #
#############

# Cleaned reference data
ref_path = os.path.join(maindir,'data/tabular/mod/dc_data/training/dc_data_reference_sampled_footprint_clean.csv')
ref = pd.read_csv(ref_path)

# Load the XGBoost classification scenario results
all_results = os.path.join(maindir,'data/tabular/mod/dc_data/results/xgboost_all_results_footprints.csv')
all_results = pd.read_csv(all_results)

# Load the image data (spectral indices and MNF transform (MNF 1&2)
stack_path = 'data/spatial/mod/dc_data/planet-data/dc_data_psscene27b.tif'
stack = rxr.open_rasterio(stack_path)
print_raster(stack)


###################################################
# Select the best performing model (by F1 or MCC) #
###################################################

# Calculating mean or median MCC for each Feature Set within each class
mean_mccs = all_results.groupby(['Feature_Set', 'Class'])['MCC'].mean().reset_index()
# Identifying the best Feature Set for each Class
best_features = mean_mccs.loc[mean_mccs.groupby('Class')['MCC'].idxmax()]
# Displaying the best models for each Class based on MCC
print("Best models for each class based on MCC:\n", best_features)

best_results = all_results[
    all_results.set_index(
        ['Feature_Set', 'Class']
    ).index.isin(best_features.set_index(['Feature_Set', 'Class']).index)]

# Create a dictionary to store the feature set bands:
feature_bands = {
    'Spectral_Bands_wIndices_Texture': ['coastal_blue','blue','green_i','green','yellow','red','rededge','nir',
                                        'ndre', 'vgnir', 'vrnir', 'ndbibg', 'ndbirg', 'corr7','corr5','corr3',
                                        'contrast3','contrast5','contrast7','idm5','idm7','idm3','asm5','asm7','asm3'],
    'Spectral_Bands': ['coastal_blue','blue','green_i','green','yellow','red','rededge','nir']
}

# Create a dictionary mapping band names to their indices using the 'long_name' attribute
bands_index = {name: i+1 for i, name in enumerate(stack.long_name)}

##################################################################################
# Run the probability creation for each class using the best performing scenario #
##################################################################################

classDict = best_features.set_index('Class')['Feature_Set'].to_dict()
print(classDict)

for cls, fs in classDict.items():
    print(f'Processing class {cls} using "{fs}"')

    # Get the band names from the feature set
    band_names = feature_bands.get(fs, [])
    band_indices = [bands_index[band] for band in band_names]
    print(band_indices)

    # Split into train and testing data for that class
    y_bin = ref['class_code'].apply(lambda x: 1 if x == cls else 0)
    X = ref[band_names]  # the 'Feature_Set'

    # Grab a weight metric
    wt = np.sum(y_bin == 0) / np.sum(y_bin == 1)  # for the 'scale_pos_weight' handling class imbalance
    print(f'Class imbalance weighting: {wt}')

    # Initialize the XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=1001,
        scale_pos_weight=wt,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    # Fit the model
    print("Fitting model ...")
    xgb_model.fit(X, y_bin)

    # Make predictions on the image data
    print("Classifying image")

    # Subset the bands based on the feature set
    stack_ = stack.sel(band=band_indices)
    print(f"Raster stack shape: {stack_.shape[1], stack_.shape[2]}")

    # Reshape the raster stack for prediction
    stack_prob = flatten_raster(stack_)

    # Fit the model the image data
    probabilities = xgb_model.predict_proba(stack_prob)[:, 1]

    # Reshape back to image
    prob_raster = probabilities.reshape(stack_.shape[1], stack_.shape[2])

    del stack_prob, probabilities

    # Create the probability raster
    print("Creating the probability raster ...")
    # Create a new DataArray for the probability raster
    prob_da = xr.DataArray(
        data=prob_raster,
        dims=["y", "x"],  # Assumed names for the spatial dimensions, adjust if necessary
        coords={
            "y": stack_["y"],  # Assuming stack_ has coordinate labels you can reuse
            "x": stack_["x"]
        },
        attrs={"long_name": "probability"}  # Set appropriate metadata
    )

    # Make sure to write the CRS
    prob_da.rio.write_crs(stack.rio.crs, inplace=True)

    # Create a new xarray DataArray from the probability raster
    output_path = os.path.join(maindir,f'data/spatial/mod/dc_data/xgb_{cls}_probability.tif')
    # Export the raster image
    prob_da.rio.to_raster(output_path)

    del stack_, prob_raster, prob_da

    print(f"Completed in {round((time.time() - begin)/60)}")



