import numpy as np
import pandas as pd

# Generate 100 samples, each with 2 features
np.random.seed(42)
X = np.random.rand(100, 2)

# Define a simple rule: Class 1 if x1 + x2 > 1, else Class 0
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Convert to a DataFrame
df = pd.DataFrame(np.column_stack((X, y)), columns=["Feature_1", "Feature_2", "Label"])

# Save to CSV
df.to_csv("binary_toy_dataset.csv", index=False)

# Display a sample
print(df.head())
