import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

class LogisticConfidence:
    def __init__(self, d, alpha=2.0):
        """
        Logistic Regression with confidence interval estimation.
        
        Parameters:
          d     - Dimensionality of the input features.
          alpha - Scaling parameter (similar to a z-value) for the confidence interval.
        """
        self.d = d
        self.alpha = alpha
        self.X = []  # List to store feature vectors
        self.y = []  # List to store labels
        self.theta_hat = np.zeros(d)  # Parameter estimate
        self.V = np.eye(d)  # Fisher information approximation (covariance matrix)

    def logistic_loss(self, theta):
        """Negative log-likelihood for logistic regression."""
        if not self.X:
            return 0
        X = np.array(self.X)
        y = np.array(self.y)
        logits = X @ theta
        probs = 1 / (1 + np.exp(-logits))
        # Add a small constant for numerical stability
        return -np.sum(y * np.log(probs + 1e-6) + (1 - y) * np.log(1 - probs + 1e-6))

    def fit(self):
        """Fit the logistic regression model (MLE) and update the covariance estimate."""
        if not self.X:
            return
        #res = minimize(self.logistic_loss, self.theta_hat, method="SLSQP")
        res = minimize(self.logistic_loss, self.theta_hat, method="BFGS")
        self.theta_hat = res.x
        X = np.array(self.X)
        logits = X @ self.theta_hat
        probs = 1 / (1 + np.exp(-logits))
        # Build the diagonal weight matrix using the logistic variance
        W = np.diag(probs * (1 - probs))
        # The Fisher information matrix approximation; add a ridge term for stability.
        self.V = X.T @ W @ X + np.eye(self.d) * 1e-6

    def update(self, x, label):
        """Add a new observation and refit the model."""
        self.X.append(x)
        self.y.append(label)
        self.fit()

    def get_confidence_interval(self, x_vec):
        """
        Compute the confidence interval for the logistic model at a given input x_vec.
        
        Returns:
          (lower_prob, upper_prob) for the predicted probability.
        """
        # Linear predictor
        linear_pred = x_vec @ self.theta_hat
        # Compute variance: x^T V^{-1} x
        V_inv = np.linalg.inv(self.V)
        variance = x_vec @ V_inv @ x_vec
        # Confidence interval on the linear predictor
        lower_lin = linear_pred - self.alpha * np.sqrt(variance)
        upper_lin = linear_pred + self.alpha * np.sqrt(variance)
        # Transform to probability space using the logistic function
        lower_prob = 1 / (1 + np.exp(-lower_lin))
        upper_prob = 1 / (1 + np.exp(-upper_lin))
        return lower_prob, upper_prob

    def get_confidence_width(self, x_vec):
        """
        Return the width of the confidence interval in probability space at x_vec.
        """
        lower_prob, upper_prob = self.get_confidence_interval(x_vec)
        return upper_prob - lower_prob

# ---------------------------------------------------------------------
# Example: Fit training data and plot a heatmap of CI width over (x1, x2)
# ---------------------------------------------------------------------

# # Suppose we have the following training data:
# # For demonstration, we create some synthetic data.
# np.random.seed(42)
# N = 100
# # Features: two-dimensional, sampled from a uniform distribution over [-2, 2]
# X_train = np.random.uniform(-2, 2, size=(N, 2))
# # True model parameters (unknown in practice)
# true_theta = np.array([1.0, -0.5])
# # Compute the probability via a logistic function of a linear combination
# logits = X_train @ true_theta
# probs = 1 / (1 + np.exp(-logits))
# # Generate binary labels
# y_train = (np.random.rand(N) < probs).astype(int)

#read my generated file


# Read the CSV file
data = pd.read_csv('/Users/annawang/Documents/Out-of-Distribution-Learning/heatmap/data_logistic.csv')

# Separate the features and labels
X_train = data[['x1', 'x2']].values  # Converts to a NumPy array of shape (n_samples, 2)
y_train = data['y'].values           # Converts to a NumPy array of shape (n_samples,)


# Create an instance of our logistic model
model = LogisticConfidence(d=2, alpha=2.0)
# Update the model with all training data
for x, label in zip(X_train, y_train):
    model.update(x, label)

# Define a grid over the feature space.
# Here we choose x1 and x2 values from -2 to 2.
grid_x1 = np.linspace(-2, 2, 100)
grid_x2 = np.linspace(-2, 2, 100)

# Create a 2D array to hold the width of the confidence intervals.
ci_width_grid = np.zeros((len(grid_x2), len(grid_x1)))  # rows: x2, columns: x1

# Loop over the grid and compute the CI width at each (x1, x2) point.
for i, x2 in enumerate(grid_x2):
    for j, x1 in enumerate(grid_x1):
        x_vec = np.array([x1, x2])
        ci_width = model.get_confidence_width(x_vec)
        ci_width_grid[i, j] = ci_width

# Plot the heatmap using matplotlib.
plt.figure(figsize=(10, 8))
im = plt.imshow(ci_width_grid, origin='lower',
                extent=[grid_x1[0], grid_x1[-1], grid_x2[0], grid_x2[-1]],
                aspect='auto', cmap='viridis')
plt.colorbar(im, label='Width of Confidence Interval')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Heatmap of Confidence Interval Width (Probability Space)')

# Overlay training data points.
# For better visualization, you can use different markers or colors based on the label.
# For example, red for label 1 and blue for label 0.
for label, marker, color in zip([0, 1], ['o', 's'], ['blue', 'red']):
    idx = np.where(y_train == label)
    print(idx)
    print(X_train[idx, 0], X_train[idx, 1])
    plt.scatter(X_train[idx, 0], X_train[idx, 1], 
                marker=marker, color=color, edgecolors='k', label=f'Label {label}', s=50, alpha =0.5)
    
# Add the decision boundary (where x1 * theta_0 + x2 * theta_1 = 0)
# Check to avoid division by zero (theta_hat[1] should not be zero)
theta_hat = model.theta_hat
if np.abs(theta_hat[1]) > 1e-6:
    # x2 = (-theta_hat[0]/theta_hat[1]) * x1
    x1_vals = np.linspace(-1, 1, 200)
    x2_vals = - (theta_hat[0] / theta_hat[1]) * x1_vals
    plt.plot(x1_vals, x2_vals, 'k--', linewidth=2, label='Decision Boundary (p=0.5)')
else:
    # If theta_hat[1] is very small, use x1 as a constant line.
    plt.axvline(x=-theta_hat[0]/1e-6, color='k', linestyle='--', linewidth=2, label='Decision Boundary')

plt.legend()

plt.show()
