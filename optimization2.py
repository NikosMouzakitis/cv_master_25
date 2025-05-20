import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load your datasets (adjust file names if needed)
train_df = pd.read_csv('features_normalized_train.csv')
test_df = pd.read_csv('features_normalized_test.csv')

# Your best 30 features from RF feature importance
top_features = [
    'abnormal_area_ratio', 'circular_area_score', 'original_glszm_GrayLevelVariance',
    'original_gldm_SmallDependenceLowGrayLevelEmphasis', 'original_glszm_SizeZoneNonUniformity',
    'mid_freq_skewness', 'original_glcm_Imc2', 'original_glszm_GrayLevelNonUniformityNormalized',
    'original_glszm_SmallAreaEmphasis', 'original_glszm_ZoneEntropy', 'high_freq_entropy',
    'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SizeZoneNonUniformityNormalized',
    'asymmetry_score', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glrlm_GrayLevelVariance',
    'asymmetry_outlier', 'original_glszm_LowGrayLevelZoneEmphasis', 'solidity_outlier',
    'original_glrlm_RunEntropy', 'original_gldm_HighGrayLevelEmphasis', 'mid_freq_entropy',
    'original_glcm_Autocorrelation', 'original_glcm_ClusterTendency', 'original_firstorder_RootMeanSquared',
    'RootMeanSquared', 'original_glcm_SumSquares', 'high_freq_skewness', 'original_gldm_GrayLevelVariance',
    'original_gldm_LargeDependenceHighGrayLevelEmphasis'
]

# Prepare train and test sets using these features
X_train = train_df[top_features].values
y_train = train_df['label'].values

X_test = test_df[top_features].values
y_test = test_df['label'].values

# Train Random Forest
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate RF
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_rf):.3f}")
print(classification_report(y_test, y_pred_rf))

# Train Neural Network (same architecture as yours)
nn = MLPClassifier(hidden_layer_sizes=(128,16), max_iter=500, activation='relu', solver='lbfgs', random_state=42)
#nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu', solver='lbfgs', random_state=42)
nn.fit(X_train, y_train)

# Predict and evaluate NN
y_pred_nn = nn.predict(X_test)
print("\nNeural Network Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nn):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_nn):.3f}")
print(classification_report(y_test, y_pred_nn))

# Plot confusion matrices side by side
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Meningioma', 'Glioma'], yticklabels=['Meningioma', 'Glioma'])
plt.title('Random Forest Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_nn), annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Meningioma', 'Glioma'], yticklabels=['Meningioma', 'Glioma'])
plt.title('Neural Network Confusion Matrix')

plt.tight_layout()
plt.show()

# ROC Curves for both
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_prob_nn = nn.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_prob_nn)

auc_rf = auc(fpr_rf, tpr_rf)
auc_nn = auc(fpr_nn, tpr_nn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='blue')
plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {auc_nn:.2f})', color='orange')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

