
"""
XGBoost classifier for predicting roof material type from satellite data

Workflow
    1. Load the reference datasets (roof material type sampled to spectral data)
    2. Identify and remove outliers from the reference data using Isolation Forests
    2. Prepare the reference data
        - By footprint centroid
        - By footprint average
    3. Split into train/test (70/30) sets
    4. Set up the binary XGBoost models for each material type with different combinations of inputs
        - Original spectral bands from PlanetScope
        - Original bands + spectral indices
        - MNF transformation of the original bands + spectral indices
    5. Test model performance and accuracy
        - Extract the decision trees, especially for the FCLS unmixing models
        - Extract the feature importance
        - Run accuracy assessment (F-score and/or MCC)

maxwell.cook@colorado.edu
"""

import os
import glob
import pandas as pd
import rioxarray as rxr
import numpy as np

from imblearn.over_sampling import SMOTE  # Import SMOTE

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

import xgboost as xgb

maindir = '/Users/max/Library/CloudStorage/OneDrive-Personal/mcook/earth-lab/opp-urban-fuels/rooftop-materials/data'

# Custom functions


# List files in a directory
def list_files(path, ext):
    return glob.glob(os.path.join(path, '**', '*{}'.format(ext)), recursive=True)


# Print raster information



# Load the training data

# Footprints
ref_tbl_path_fp = os.path.join(maindir,'tabular/mod/dc_data/training/dc_data_reference_sampled_footprint.csv')
# Centroids
ref_tbl_path = os.path.join(maindir,'tabular/mod/dc_data/training/dc_data_reference_sampled.csv')


####################################
# Load and prep the reference data #
####################################

# Load the reference datasets and filter

ref_list = [ref_tbl_path, ref_tbl_path_fp]  # testing both the centroid and footprint-based samples
names = ['centroid','footprint']

refs_out = []
for i in range(len(ref_list)):
    print(names[i])

    # Original spectral bands from PlanetScope SuperDove + Spectral Indices + MNF1
    ref = pd.read_csv(ref_list[i])
    print(ref.columns)
    print(ref['class_code'].value_counts())
    print(ref.columns)

    try:
        ref = ref.drop(['Unnamed: 0','index','areaUTMsqft','description'], axis=1)
    except KeyError:
        print("Missing key")
        ref = ref.drop(['Unnamed: 0','areaUTMsqft','description','geometry'], axis=1)

    print(list(ref.columns))

    # Merge the shingle classes (wood shingle and shingle)
    merge = {'WS': 'WSH', 'SH': 'WSH'}
    ref['class_code'].replace(merge, inplace=True)
    ref['class_code'].value_counts()  # check the counts

    # Filter out likely vegetated samples (Normalized Difference Red-edge Index > 0.42)
    ref = ref[ref['ndre'] < 0.42]  # filter out high NDRE values (vegetation)

    # Handle outliers using anomaly detection via Isolation Forest model
    # Loop through classes, identify outliers
    refs = []
    for cls in ref['class_code'].unique():
        print(f"Processing outliers for class {cls}")

        df = ref[ref['class_code'] == cls]  # filter to the class
        df_num = df.drop(['class_code','uid'], axis=1)  # keep numerical columns

        # Initialize the Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=1001,
            max_samples='auto',
            contamination=float(0.05),  # expected proportion of outliers
            random_state=42
        )

        # Predict the outliers / anomalies
        preds = iso_forest.fit_predict(df_num)

        # Selecting data without outliers
        df_ = df[preds != -1]
        # Append the DataFrame without outliers to the list
        refs.append(df_)

        del df, df_num, preds, df_

    # Concatenate all the DataFrames in the list into a single DataFrame
    ref = pd.concat(refs)
    print(ref['class_code'].value_counts())  # check the counts again

    del refs, iso_forest  # clean up

    # Save this out to a file
    out_file = os.path.join(maindir,f'tabular/mod/dc_data/training/dc_data_reference_sampled_{names[i]}_clean.csv')
    ref.to_csv(out_file)

    refs_out.append(ref)

    del ref


# ~~~~~~~~~~~~~~~Set up the model dataframes~~~~~~~~~~~~~~~~ #

# ref = refs_out[1]  # this is the footprint (mean) reference data
sources = ["centroid","footprints"]

for ii in range(len(refs_out)):
    ref = refs_out[ii]
    source = sources[ii]

    # Get the unique classes and set a numeric class code for modelling
    classes = {code: idx for idx, code in enumerate(ref['class_code'].unique())}  # create numeric label

    # First model (original spectral bands)
    ref_m1 = ref[['class_code','coastal_blue','blue','green_i','green','yellow','red','rededge','nir']]
    ref_m1['code'] = ref_m1['class_code'].apply(lambda x: classes.get(x, x))
    print(f'First model features: {list(ref_m1.columns)}')

    # Second model (original spectral bands + spectral indices)
    ref_m2 = ref[
        ['class_code','coastal_blue','blue','green_i','green','yellow','red','rededge','nir',
         'ndre', 'vgnir', 'vrnir', 'ndbibg', 'ndbirg']
    ]
    ref_m2['code'] = ref_m2['class_code'].apply(lambda x: classes.get(x, x))
    print(f'Second model features: {list(ref_m2.columns)}')

    # Third model (Spectral Indices + MNF1)
    ref_m3 = ref[
        ['class_code', 'ndre', 'vgnir', 'vrnir', 'ndbibg', 'ndbirg', 'mnf1']
    ]
    ref_m3['code'] = ref_m3['class_code'].apply(lambda x: classes.get(x, x))
    print(f'Third model features: {list(ref_m3.columns)}')

    # Fourth model (red band textural features)\
    ref_m4 = ref[['class_code','corr7','corr5','corr3','contrast3','contrast5',
                  'contrast7','idm5','idm7','idm3','asm5','asm7','asm3']]
    ref_m4['code'] = ref_m4['class_code'].apply(lambda x: classes.get(x, x))
    print(f'Fourth model features: {list(ref_m4.columns)}')

    # Fifth model (MNF1 + spectral indices + textural features)
    ref_m5 = ref[['class_code','ndre', 'vgnir', 'vrnir', 'ndbibg', 'ndbirg', 'mnf1',
                  'corr7','corr5','corr3','contrast3','contrast5',
                  'contrast7','idm5','idm7','idm3','asm5','asm7','asm3']]
    ref_m5['code'] = ref_m5['class_code'].apply(lambda x: classes.get(x, x))
    print(f'Fourth model features: {list(ref_m5.columns)}')

    # 6th model (original bands + spectral indices + textural features)
    ref_m6 = ref[['class_code','coastal_blue','blue','green_i','green','yellow','red','rededge','nir',
                  'ndre', 'vgnir', 'vrnir', 'ndbibg', 'ndbirg',
                  'corr7','corr5','corr3','contrast3','contrast5',
                  'contrast7','idm5','idm7','idm3','asm5','asm7','asm3']]
    ref_m6['code'] = ref_m6['class_code'].apply(lambda x: classes.get(x, x))
    print(f'Fourth model features: {list(ref_m6.columns)}')

    del ref  # clean up

    # Combine the reference datasets into a dictionary
    ref_dict = {
        'Spectral_Bands': ref_m1,
        'Spectral_Bands_wIndices': ref_m2,
        'Indices_wMNF': ref_m3,
        'Texture_Bands': ref_m4,
        'Indices_wMNF_wTexture': ref_m5,
        'Spectral_Bands_wIndices_Texture': ref_m6
    }

    # Grab a dictionary of the class name / class numeric code for reference later
    class_dict = dict(zip(ref_m1['code'], ref_m1['class_code']))
    print(class_dict)

    # Check the sample size per class
    class_n = np.bincount(ref_m1['code'])  # get the class counts
    for mat, count in enumerate(class_n):
        print(f"'{mat}': {count} samples")

    #################################
    # Set up the XGBoost classifier #
    #################################

    class_list = ref_m1['code'].unique()

    n_folds = 10  # number of folds to run for cross-validation

    all_results = pd.DataFrame()  # to store the model performance metrics (default 0.50 cutoff)
    all_feat_imps = pd.DataFrame()  # to store the feature importances
    all_prob_preds = pd.DataFrame()  # for testing optimum cutoff

    # Loop through the classification scenarios, run the 10-fold CV XGBoost
    for ft_name, ft_data in ref_dict.items():
        print(f'Running XGBoost for reference set: {ft_name}')

        # Set up the model data
        y = ft_data['code']
        X = ft_data.drop(['class_code','code'], axis=1)

        # Define dataframes to store results for this feature set
        results = pd.DataFrame()  # to store the model performance metrics
        feat_imps = pd.DataFrame()  # to store the feature importances
        prob_preds = pd.DataFrame()  # for testing optimum cutoff

        for cls in class_list:
            print(f'Processing class: {class_dict.get(cls)}')

            # Create a binary reference dataset and grab the weights
            y_bin = y.apply(lambda x: 1 if x == cls else 0)
            wt = np.sum(y_bin == 0) / np.sum(y_bin == 1)  # for the 'scale_pos_weight' handling class imbalance
            print(f'Class imbalance weighting: {wt}')

            # Set up the stratified K-fold
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            # Loop the folds
            fold_idx = 1
            for train_index, test_index in skf.split(X, y_bin):
                print(f'Fold: {fold_idx}')

                # Split into train/test sets
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y_bin.iloc[train_index], y_bin.iloc[test_index]

                # Check on class imbalance, perform SMOTE if necessary
                if wt > 10:  # 10:1 threshold for class imbalance
                    print("Performing SMOTE for class imbalance ...")
                    # Apply SMOTE
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)

                    # Initialize the XGBoost classifier
                    # scale_pos_weight not used for high class imbalance
                    xgb_model = xgb.XGBClassifier(
                        objective='binary:logistic',
                        n_estimators=101,
                        learning_rate=0.1,
                        max_depth=8,
                        random_state=42
                    )

                else:
                    # Initialize the XGBoost classifier without SMOTE
                    xgb_model = xgb.XGBClassifier(
                        objective='binary:logistic',
                        n_estimators=101,
                        scale_pos_weight=wt,
                        learning_rate=0.1,
                        max_depth=8,
                        random_state=42
                    )

                # Fit the model
                xgb_model.fit(X_train, y_train)

                # Store feature importance
                fold_imps = pd.DataFrame({
                    'Feature_Set': ft_name,
                    'Class': class_dict.get(cls),
                    'Fold': fold_idx,
                    'Feature': X.columns,
                    'Importance': xgb_model.feature_importances_})

                feat_imps = pd.concat([feat_imps, fold_imps], axis=0)

                # Predict on the test set
                y_pred = xgb_model.predict(X_test)
                # print("Unique predictions:", np.unique(y_pred))

                # Retrieve the accuracy/performance metrics
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)

                # Store the metrics into the results data frame
                fold_results = pd.DataFrame({
                    'Feature_Set': [ft_name],
                    'Class': [class_dict.get(cls)],
                    'Fold': [fold_idx],
                    'Precision': [precision],
                    'Recall': [recall],
                    'F1': [f1],
                    'MCC': [mcc]
                })
                results = pd.concat([results, fold_results], ignore_index=True)

                # Store the probability values for cutoff testing
                y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

                # Store probabilities and true labels
                fold_probs = pd.DataFrame({
                    'Feature_Set': ft_name,
                    'TrueLabel': y_test,
                    'PredictedProb': y_pred_proba,
                    'Class': class_dict.get(cls),
                    'Fold': fold_idx
                })
                prob_preds = pd.concat([prob_preds, fold_probs], ignore_index=True)

                fold_idx += 1

                del fold_probs, fold_results, fold_imps

        # Append the feature set-specific results to the overall results dataframes
        all_results = pd.concat([all_results, results], ignore_index=True)
        all_feat_imps = pd.concat([all_feat_imps, feat_imps], ignore_index=True)
        all_prob_preds = pd.concat([all_prob_preds, prob_preds], ignore_index=True)

        del results, feat_imps, prob_preds

    # Save the tables to CSV files
    all_results.to_csv(os.path.join(maindir,f'tabular/mod/dc_data/results/xgboost_all_results_{source}.csv'))
    all_feat_imps.to_csv(os.path.join(maindir,f'tabular/mod/dc_data/results/xgboost_all_ftr_imps_{source}.csv'))
    all_prob_preds.to_csv(os.path.join(maindir,f'tabular/mod/dc_data/results/xgboost_all_prob_preds_{source}.csv'))