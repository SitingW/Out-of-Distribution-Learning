import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples per cluster
n_samples = 500

# Define cluster centers inside the unit ball
center_1 = np.array([0.5, 0.5])  # Cluster for y=1
center_0 = np.array([-0.5, -0.5])  # Cluster for y=0

# Define spread (standard deviation of Gaussian)
spread = 0.2

# Generate points for cluster 1 (y=1)
x1_cluster1 = np.random.normal(center_1[0], spread, n_samples)
x2_cluster1 = np.random.normal(center_1[1], spread, n_samples)
X1 = np.column_stack((x1_cluster1, x2_cluster1))
y1 = np.ones(n_samples)

# Generate points for cluster 0 (y=0)
x1_cluster0 = np.random.normal(center_0[0], spread, n_samples)
x2_cluster0 = np.random.normal(center_0[1], spread, n_samples)
X0 = np.column_stack((x1_cluster0, x2_cluster0))
y0 = np.zeros(n_samples)

# Combine both clusters
X = np.vstack((X1, X0))
y = np.concatenate((y1, y0))

# Ensure all points are within the unit ball
mask = np.linalg.norm(X, axis=1) <= 1
X, y = X[mask], y[mask]  # Keep only valid points

# Convert to a DataFrame
df = pd.DataFrame(np.column_stack((X, y)), columns=["x1", "x2", "y"])

# Save to CSV
df.to_csv("data_logistic.csv", index=False)
