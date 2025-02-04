import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class LogisticUCB:
    def __init__(self, d, alpha=2.0):
        """
        Logistic Bandit with UCB using a GLM.
        
        Parameters:
        d     - Feature dimension
        alpha - Exploration/confidence scaling parameter
        """
        self.d = d
        self.alpha = alpha
        self.X = []  # Collected feature vectors from past pulls
        self.y = []  # Collected outcomes
        self.theta_hat = np.zeros(d)  # Parameter estimate
        self.V = np.eye(d)  # Fisher Information approximation (covariance matrix)

    def logistic_loss(self, theta):
        """Negative log-likelihood for logistic regression."""
        if not self.X:
            return 0  # No data available yet
        X = np.array(self.X)
        y = np.array(self.y)
        logits = X @ theta
        probs = 1 / (1 + np.exp(-logits))
        # Adding a small constant (1e-6) for numerical stability
        return -np.sum(y * np.log(probs + 1e-6) + (1 - y) * np.log(1 - probs + 1e-6))

    def fit(self):
        """Fit the logistic regression model (MLE) and update the covariance estimate."""
        if not self.X:
            return
        res = minimize(self.logistic_loss, self.theta_hat, method="BFGS")
        self.theta_hat = res.x
        X = np.array(self.X)
        logits = X @ self.theta_hat
        probs = 1 / (1 + np.exp(-logits))
        # Diagonal weight matrix using the logistic variance function
        W = np.diag(probs * (1 - probs))
        # Update Fisher Information matrix with a small ridge term for numerical stability
        self.V = X.T @ W @ X + np.eye(self.d) * 1e-6

    def update(self, x_k, reward):
        """Update the model with a new observation and refit."""
        self.X.append(x_k)
        self.y.append(reward)
        self.fit()

    def get_confidence_set(self, grid):
        """
        For each point in 'grid', compute the confidence interval for the linear predictor,
        and optionally, for the probability (after applying the logistic function).
        Returns a dictionary mapping each grid point to its (lower, upper) interval.
        """
        confidence_intervals = {}
        # Compute the inverse of V once (if V is well-conditioned)
        V_inv = np.linalg.inv(self.V)
        for x_val in grid:
            # For a 1D feature, treat x_val as a 1-dimensional vector.
            x_vec = np.array([x_val])
            # Compute the linear predictor
            linear_pred = x_vec @ self.theta_hat
            # Variance of the linear predictor: x^T * V_inv * x
            variance = x_vec @ V_inv @ x_vec
            # Confidence interval on the linear predictor
            lower = linear_pred - self.alpha * np.sqrt(variance)
            upper = linear_pred + self.alpha * np.sqrt(variance)
            # Optionally, transform to probability space using the logistic function
            lower_prob = 1 / (1 + np.exp(-lower))
            upper_prob = 1 / (1 + np.exp(-upper))
            confidence_intervals[x_val] = (lower_prob.item(), upper_prob.item())
        return confidence_intervals

# --------------------
# Example usage:
# --------------------
np.random.seed(42)
d = 1  # Feature dimension is 1 for this example
bandit = LogisticUCB(d, alpha=2.0)

# Simulate some data updates
# Assume true_theta is unknown; here we use a fixed one for simulation
true_theta = np.array([1.0])  # True parameter for simulation purposes
num_updates = 50
for _ in range(num_updates):
    # Sample a random feature from uniform(-1, 1)
    x_sample = np.array([np.random.uniform(-1, 1)])
    # Generate reward according to logistic probability
    prob = 1 / (1 + np.exp(-x_sample @ true_theta))
    reward = 1 if np.random.rand() < prob else 0
    bandit.update(x_sample, reward)

# Define the grid from -1 to 1 with a step of 0.1
grid = np.arange(-1, 1.1, 0.1)
confidence_intervals = bandit.get_confidence_set(grid)

# Compute the width of the confidence intervals for each grid point
ci_widths = []
for x_val in grid:
    lower, upper = confidence_intervals[x_val]
    ci_widths.append(upper - lower)

# Create a scatter plot where the color represents the CI width
plt.figure(figsize=(8, 4))
sc = plt.scatter(grid, ci_widths, c=ci_widths, cmap='viridis', s=100)
plt.colorbar(sc, label='Width of Confidence Interval')
plt.xlabel('x')
plt.ylabel('Confidence Interval Width')
plt.title('Heatmap of Confidence Interval Width Across x')
plt.show()