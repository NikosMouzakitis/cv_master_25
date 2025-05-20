# Configuration
SELECTED_FEATURES = [
        'original_firstorder_10Percentile', 
        'original_firstorder_90Percentile', 
        'original_firstorder_Energy', 
        'original_firstorder_Entropy', 
        'original_firstorder_InterquartileRange', 
        'original_firstorder_Kurtosis', 
        'original_firstorder_Maximum',
        'original_firstorder_MeanAbsoluteDeviation',
        'original_firstorder_Mean',
        'original_firstorder_Median',
        'original_firstorder_Minimum',
        'original_firstorder_Range',
        'original_firstorder_RobustMeanAbsoluteDeviation',
        'original_firstorder_RootMeanSquared',
        'original_firstorder_Skewness',
        'original_firstorder_TotalEnergy',
        'original_firstorder_Uniformity',
        'original_firstorder_Variance',
        'original_glcm_Autocorrelation',
        'original_glcm_ClusterProminence',
        'original_glcm_ClusterShade',
        'original_glcm_ClusterTendency',
        'original_glcm_Contrast',
        'original_glcm_Correlation',
        'original_glcm_DifferenceAverage',
        'original_glcm_DifferenceEntropy',
        'original_glcm_DifferenceVariance',
        'original_glcm_Id',
        'original_glcm_Idm',
        'original_glcm_Idmn',
        'original_glcm_Idn',
        'original_glcm_Imc1',
        'original_glcm_Imc2',
        'original_glcm_InverseVariance',
        'original_glcm_JointAverage',
        'original_glcm_JointEnergy',
        'original_glcm_JointEntropy',
        'original_glcm_MCC',
        'original_glcm_MaximumProbability',
        'original_glcm_SumAverage',
        'original_glcm_SumEntropy',
        'original_glcm_SumSquares',
        'original_gldm_DependenceEntropy',
        'original_gldm_DependenceNonUniformity',
        'original_gldm_DependenceNonUniformityNormalized',
        'original_gldm_DependenceVariance',
        'original_gldm_GrayLevelNonUniformity',
        'original_gldm_GrayLevelVariance',
        'original_gldm_HighGrayLevelEmphasis',
        'original_gldm_LargeDependenceEmphasis',
        'original_gldm_LargeDependenceHighGrayLevelEmphasis',
        'original_gldm_LargeDependenceLowGrayLevelEmphasis',
        'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis',
        'original_gldm_SmallDependenceHighGrayLevelEmphasis',
        'original_gldm_SmallDependenceLowGrayLevelEmphasis',
        'original_glrlm_GrayLevelNonUniformity',
        'original_glrlm_GrayLevelNonUniformityNormalized',
        'original_glrlm_GrayLevelVariance',
        'original_glrlm_HighGrayLevelRunEmphasis',
        'original_glrlm_LongRunEmphasis',
        'original_glrlm_LongRunHighGrayLevelEmphasis',
        'original_glrlm_LongRunLowGrayLevelEmphasis',
        'original_glrlm_LowGrayLevelRunEmphasis',
        'original_glrlm_RunEntropy',
        'original_glrlm_RunLengthNonUniformity',
        'original_glrlm_RunLengthNonUniformityNormalized',
        'original_glrlm_RunPercentage',
        'original_glrlm_RunVariance',
        'original_glrlm_ShortRunEmphasis',
        'original_glrlm_ShortRunHighGrayLevelEmphasis',
        'original_glrlm_ShortRunLowGrayLevelEmphasis',
        'original_glszm_GrayLevelNonUniformity',
        'original_glszm_GrayLevelNonUniformityNormalized',
        'original_glszm_GrayLevelVariance',
        'original_glszm_HighGrayLevelZoneEmphasis',
        'original_glszm_LargeAreaEmphasis',
        'original_glszm_LargeAreaHighGrayLevelEmphasis',
        'original_glszm_LargeAreaLowGrayLevelEmphasis',
        'original_glszm_LowGrayLevelZoneEmphasis',
        'original_glszm_SizeZoneNonUniformity',
        'original_glszm_SizeZoneNonUniformityNormalized',
        'original_glszm_SmallAreaEmphasis',
        'original_glszm_SmallAreaHighGrayLevelEmphasis',
        'original_glszm_SmallAreaLowGrayLevelEmphasis',
        'original_glszm_ZoneEntropy',
        'original_glszm_ZonePercentage',
        'original_glszm_ZoneVariance',
        'original_ngtdm_Busyness',
        'original_ngtdm_Coarseness',
        'original_ngtdm_Complexity',
        'original_ngtdm_Contrast',
        'original_ngtdm_Strength',
        '10Percentile', 
        '90Percentile',
        'Energy',
        'Entropy',
        'InterquartileRange',
        'Kurtosis',
        'Maximum',
        'MeanAbsoluteDeviation',
        'Mean',
        'Median',
        'Minimum',
        'Range',
        'RobustMeanAbsoluteDeviation',
        'RootMeanSquared',
        'Skewness',
        'TotalEnergy',
        'Uniformity',
        'Variance',
        'intensity_skewness',
        'intensity_outlier_score',
        'high_intensity_area',
        'max_circularity',
        'top3_circularity_mean',    
        'solidity_outlier',
        'abnormal_area_ratio',
        'circular_area_score',
        'asymmetry_score',
        'asymmetry_outlier',
        'boundary_sharpness_mean',
        'boundary_sharpness_max',
        'boundary_sharpness_outlier',
        'low_freq_energy',
        'low_freq_entropy',
        'low_freq_mean',
        'low_freq_std',
        'low_freq_skewness',
        'mid_freq_energy',
        'mid_freq_entropy',
        'mid_freq_mean',
        'mid_freq_std',
        'mid_freq_skewness',
        'high_freq_energy',
        'high_freq_entropy',
        'high_freq_mean',
        'high_freq_std',
        'high_freq_skewness'
       ]


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load your datasets
train_df = pd.read_csv('features_normalized_train.csv')
test_df = pd.read_csv('features_normalized_test.csv')

X_train = train_df[SELECTED_FEATURES].values
y_train = train_df['label'].values
X_test = test_df[SELECTED_FEATURES].values
y_test = test_df['label'].values

# Step 1: Quick RF hyperparameter tuning
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [0.5]
}

print("\nStarting RandomizedSearchCV for RF tuning...")
base_rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(base_rf, param_distributions=param_dist,
                                   n_iter=10, cv=3, n_jobs=-1, verbose=1,
                                   scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
print("\nBest RF parameters found:", random_search.best_params_)

# Step 2: Select top 30 features by importance from tuned RF
importances = best_rf.feature_importances_
top_indices = np.argsort(importances)[-30:]  # indices of top 30 features (lowest to highest)
top_features = [SELECTED_FEATURES[i] for i in top_indices]

print("\nTop 30 features selected:")
print(top_features)

X_train_top = train_df[top_features].values
X_test_top = test_df[top_features].values

# Step 3: Retrain RF with only top 30 features
best_rf.fit(X_train_top, y_train)

# Predict and evaluate
y_pred_rf = best_rf.predict(X_test_top)
print("\nRandom Forest Results on top 30 features:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_rf):.3f}")
print(classification_report(y_test, y_pred_rf))

# Step 4: Neural Network training - keep as is
print("\nTraining Neural Network...")
nn = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    random_state=42
)
nn.fit(X_train, y_train)

y_pred_nn = nn.predict(X_test)
print("\nNeural Network Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nn):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_nn):.3f}")
print(classification_report(y_test, y_pred_nn))

# Step 5: Plot confusion matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_rf),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Meningioma', 'Glioma'],
            yticklabels=['Meningioma', 'Glioma'])
plt.title('Random Forest Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_nn),
            annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Meningioma', 'Glioma'],
            yticklabels=['Meningioma', 'Glioma'])
plt.title('Neural Network Confusion Matrix')

plt.tight_layout()
plt.show()

# Step 6: Plot ROC curves
y_prob_rf = best_rf.predict_proba(X_test_top)[:, 1]
y_prob_nn = nn.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_prob_nn)

auc_rf = auc(fpr_rf, tpr_rf)
auc_nn = auc(fpr_nn, tpr_nn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='blue')
plt.plot(fpr_nn, tpr_nn, label=f'Neural Net (AUC = {auc_nn:.2f})', color='orange')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

