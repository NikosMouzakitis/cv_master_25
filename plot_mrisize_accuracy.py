import matplotlib.pyplot as plt
import numpy as np

# Classic Octave/Matlab style
plt.style.use('classic')

# Data
mri_sizes = np.array([100, 200, 400, 500, 800, 1600, 2000])
rf_scores = np.array([85, 85.5, 90, 92, 86.9, 91.9, 92])
mlp_scores = np.array([87.5, 87.5, 91.2, 87, 88.1, 90, 91.5])

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(mri_sizes, rf_scores, 'bo-', label='Random Forest')
plt.plot(mri_sizes, mlp_scores, 'rs--', label='MLP-NN')

# Labels and title
plt.xlabel('MRI Dataset Size (Train & Test)', fontsize=12)
plt.ylabel('AUC (%)', fontsize=12)
plt.title('Model AUC vs MRI Dataset Size', fontsize=14)

# Limits and ticks
plt.ylim(0, 100)
plt.xticks(mri_sizes)
plt.yticks(np.arange(0, 101, 10))

# Grid and legend
plt.grid(True, linestyle=':', linewidth=0.7)
plt.legend(loc='lower left', fontsize=13)  # << Legend in southwest

plt.tight_layout()
plt.show()

