
"""
Load the XGBoost results for post-processing and plotting
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

maindir = '/Users/max/Library/CloudStorage/OneDrive-Personal/mcook/earth-lab/opp-urban-fuels/rooftop-materials'
datatab = os.path.join(maindir,'data/tabular/mod')

source = "footprints"  # use the centroid-based sampling

# Load the XGBoost classification scenario results
all_results = os.path.join(datatab,f'dc_data/results/xgboost_all_results_{source}.csv')
all_feat_imps = os.path.join(datatab,f'dc_data/results/xgboost_all_ftr_imps_{source}.csv')
all_prob_preds = os.path.join(datatab,f'dc_data/results/xgboost_all_prob_preds_{source}.csv')

all_results = pd.read_csv(all_results)
all_feat_imps = pd.read_csv(all_feat_imps)
all_prob_preds = pd.read_csv(all_prob_preds)


###################################################################
# Metrics plots based on the default classification cutoff (0.50) #
###################################################################

# Plot the results across folds using boxplots

# Ensure 'Class' and 'Feature_Set' are categorical
all_results['Class'] = all_results['Class'].astype('category')
all_results['Feature_Set'] = all_results['Feature_Set'].astype('category')

# Prep the accuracy metrics for facet wrapping
results_m = pd.melt(
    all_results,
    id_vars=['Class', 'Fold', 'Feature_Set'],
    var_name='Metric',
    value_name='Score'
)

# Grab just the metrics we care about for now
results_m = results_m[results_m['Metric'].isin(['F1', 'MCC'])]

# Create a figure and axes
fig, axes = plt.subplots(1, 1, figsize=(6, 6))  # Adjust figure size as needed
# Plot MCC score across models
sns.boxplot(x='Class', y='Score', hue='Feature_Set', data=results_m[results_m['Metric'] == 'MCC'], ax=axes)
axes.set_xlabel('Class')
axes.set_ylabel('MCC')
axes.tick_params(axis='x', rotation=45)
# Adjust layout for readability
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust top spacing for the title
# Optional: Add a legend
axes.legend(loc='upper right')

plt.savefig('figures/FigX_class_scenarios_gt50.png', dpi=300, bbox_inches='tight')


###################################################
# Select the best performing model (by F1 or MCC) #

# Calculating mean or median MCC for each Feature Set within each class
mean_mccs = all_results.groupby(['Feature_Set', 'Class'])['MCC'].mean().reset_index()
# Identifying the best Feature Set for each Class
best_features = mean_mccs.loc[mean_mccs.groupby('Class')['MCC'].idxmax()]
# Displaying the best models for each Class based on MCC
print("Best models for each class based on MCC:\n", best_features)

# Extract results for the best models for further analysis
best_results = all_results[
    all_results.set_index(
        ['Feature_Set', 'Class']
    ).index.isin(best_features.set_index(['Feature_Set', 'Class']).index)]
best_feat_imps = all_feat_imps[
    all_feat_imps.set_index(
        ['Feature_Set', 'Class']
    ).index.isin(best_features.set_index(['Feature_Set', 'Class']).index)]
best_probs = all_prob_preds[
    all_prob_preds.set_index(
        ['Feature_Set', 'Class']
    ).index.isin(best_features.set_index(['Feature_Set', 'Class']).index)]


##########################################################
# Identify the optimal cutoff value for each class/model #
##########################################################

# First, identify the optimum cutoff for each model

# Grab a dictionary of the class name / class numeric code for reference later
class_list = all_results['Class'].unique()

cutoffs = np.linspace(0, 1, num=20)  # 10 cutoffs from 0 to 1
opt_cuts = pd.DataFrame()
accmeas = pd.DataFrame()

n_folds = 10

# Iterate through each class
for cls in class_list:
    class_metrics = []

    for c in cutoffs:
        # Filter the probabilities for the current class
        fold_probs = best_probs[best_probs['Class'] == cls]

        # Apply the cutoff to get predictions
        y_pred = (fold_probs['PredictedProb'] >= c).astype(int)

        # Ensure that both y_pred and fold_probs['TrueLabel'] are of the same length
        assert len(y_pred) == len(fold_probs['TrueLabel']), "Length mismatch between y_pred and TrueLabel"

        # Calculate metrics
        # f1 = f1_score(fold_probs['TrueLabel'], y_pred, zero_division=0)
        mcc = matthews_corrcoef(fold_probs['TrueLabel'], y_pred)

        class_metrics.append({
            'Class': cls,
            'Cutoff': c,
            # 'F1': f1,
            'MCC': mcc
        })

    # Convert to DataFrame
    class_metrics_df = pd.DataFrame(class_metrics)

    # Find the cutoff with the highest average MCC
    best_cutoff = class_metrics_df.loc[class_metrics_df['MCC'].idxmax()]
    opt_cuts = pd.concat([opt_cuts, pd.DataFrame([best_cutoff])], ignore_index=True)

print(opt_cuts)


# Second, recalculate the accuracy metrics at the optimum threshold

# Initialize a DataFrame to store the recalculated metrics
opt_mets = pd.DataFrame()

# Iterate through each class and fold, using the optimal cutoff found
for cls in class_list:
    # Get the optimal cutoff for the current class
    thresh_opt = opt_cuts.loc[opt_cuts['Class'] == cls, 'Cutoff'].values[0]

    for fold in range(1, n_folds + 1):
        # Filter the probabilities and true labels for the current class and fold
        fold_probs = best_probs[(best_probs['Class'] == cls) & (best_probs['Fold'] == fold)]
        # Apply the optimal cutoff
        y_pred_optimal = (fold_probs['PredictedProb'] >= thresh_opt).astype(int)

        # Recalculate metrics
        precision_opt = precision_score(fold_probs['TrueLabel'], y_pred_optimal, zero_division=0)
        recall_opt = recall_score(fold_probs['TrueLabel'], y_pred_optimal, zero_division=0)
        f1_opt = f1_score(fold_probs['TrueLabel'], y_pred_optimal)
        mcc_opt = matthews_corrcoef(fold_probs['TrueLabel'], y_pred_optimal)

        # Store the recalculated metrics
        opt_mets_df = pd.DataFrame({
            'Class': [cls],
            'Fold': [fold],
            'Optimal_cutoff': [thresh_opt],
            # 'Accuracy': [accuracy_opt],
            'Precision': [precision_opt],
            'Recall': [recall_opt],
            'F1': [f1_opt],
            'MCC': [mcc_opt]
        })

        opt_mets = pd.concat([opt_mets, opt_mets_df], ignore_index=True)

print(opt_mets)

# Group by 'Class' and calculate the mean of each metric
class_avg_mets = opt_mets.groupby('Class').mean()
# Reset index to make 'Class' a column again, if needed
class_avg_mets.reset_index(inplace=True)
class_avg_mets = class_avg_mets[['Class','Optimal_cutoff','Precision','Recall','F1','MCC']]
print(class_avg_mets)

# Save out
class_avg_mets.to_csv('data/tabular/mod/dc_data/results/xgboost_opt_cutoff_avg_accmets.csv')


####################################################
# Plot the new boxplot based on the optimum cutoff #
####################################################

# Melt the DataFrame to make it suitable for Seaborn's boxplot
opt_mets_m = pd.melt(opt_mets, id_vars=['Class', 'Fold', 'Optimal_cutoff'], var_name='Metric', value_name='Score')
# Convert 'Class' to a categorical type for proper ordering in the plot
opt_mets_m['Class'] = opt_mets_m['Class'].astype('category')

# Plot just the MCC and create a table with summary results

# Create a figure and axes
fig, axes = plt.subplots(1, 1, figsize=(6, 6))  # Adjust figure size as needed

sns.boxplot(x='Class', y='Score', data=opt_mets_m[opt_mets_m['Metric'] == 'MCC'], ax=axes)
axes.set_xlabel('Class')
axes.set_ylabel('MCC')
axes.tick_params(axis='x', rotation=45)
# Adjust layout for readability
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust top spacing for the title

plt.savefig('figures/FigX_class_scenarios_opt.png', dpi=300, bbox_inches='tight')


##############################################
# ROC and PR curves using the optimal cutoff #
##############################################

# Prepare to store ROC and PR curve data
roc_data = {}
pr_data = {}

for cls in class_list:
    cls_probs = best_probs[best_probs['Class'] == cls]

    # ROC Curve
    fpr, tpr, roc_cutoffs = roc_curve(cls_probs['TrueLabel'], cls_probs['PredictedProb'])
    roc_auc = auc(fpr, tpr)
    roc_data[cls] = (fpr, tpr, roc_auc, roc_cutoffs)

    # PR Curve
    precision, recall, pr_cutoffs = precision_recall_curve(cls_probs['TrueLabel'], cls_probs['PredictedProb'])
    pr_auc = average_precision_score(cls_probs['TrueLabel'], cls_probs['PredictedProb'])
    pr_data[cls] = (precision, recall, pr_auc, pr_cutoffs)

# Plot ROC and PR curves
# Calculate the number of rows needed for the subplot (each class gets a row)
num_rows = len(class_list)
# Set up the matplotlib figure with a certain size
fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(8.5, 12))
# Iterate over each class to plot in the grid
for idx, (cls, data) in enumerate(roc_data.items()):
    # Get the optimum cutoff value
    opt_cut_class = class_avg_mets[class_avg_mets['Class'] == cls]['Optimal_cutoff'].iloc[0]
    print(opt_cut_class)

    # ROC Curve
    fpr, tpr, roc_auc, roc_cutoffs = roc_data[cls]
    roc_index = np.argmin(np.abs(roc_cutoffs - opt_cut_class))

    axes[idx, 0].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[idx, 0].scatter(fpr[roc_index], tpr[roc_index], color='red', label=f'Optimal Cutoff = {opt_cut_class:.2f}')
    axes[idx, 0].plot([0, 1], [0, 1], 'k--')
    axes[idx, 0].set_xlabel('False Positive Rate')
    axes[idx, 0].set_ylabel('True Positive Rate')
    axes[idx, 0].set_title(f'ROC Curve for Class {cls}')
    axes[idx, 0].legend(loc="lower right", fontsize='small', title_fontsize='small')

    # PR Curve
    precision, recall, pr_auc, pr_cutoffs = pr_data[cls]
    pr_index = np.argmin(np.abs(pr_cutoffs - opt_cut_class))

    axes[idx, 1].plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    if pr_index < len(recall):  # Check to prevent index error
        axes[idx, 1].scatter(recall[pr_index], precision[pr_index], color='red',
                             label=f'Optimal Cutoff = {opt_cut_class:.2f}')
    axes[idx, 1].set_xlabel('Recall')
    axes[idx, 1].set_ylabel('Precision')
    axes[idx, 1].set_title(f'Precision-Recall Curve for Class {cls}')
    axes[idx, 1].legend(loc="lower left", fontsize='small', title_fontsize='small')

# Adjust layout for readability
fig.tight_layout()

plt.savefig('figures/FigX_PR_ROC_curves.png', dpi=300, bbox_inches='tight')
