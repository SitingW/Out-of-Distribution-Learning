import numpy as np
from scipy.optimize import minimize

class LogisticUCB:
    def __init__(self, d, alpha=2.0):
        """
        Logistic Bandit with UCB.
        
        Parameters:
        d     - Feature dimension
        alpha - Exploration parameter
        """
        self.d = d
        self.alpha = alpha
        self.X = []  # Feature vectors of pulled arms
        self.y = []  # Corresponding rewards
        self.theta_hat = np.zeros(d)  # Parameter estimate
        self.V = np.eye(d)  # Covariance matrix (Hessian approximation)

    def logistic_loss(self, theta):
        """Negative log-likelihood for logistic regression."""
        if not self.X:
            return 0  # No data yet
        X = np.array(self.X)
        y = np.array(self.y)
        logits = X @ theta
        probs = 1 / (1 + np.exp(-logits))
        return -np.sum(y * np.log(probs + 1e-6) + (1 - y) * np.log(1 - probs + 1e-6))

    def fit(self):
        """Fit the logistic regression model using MLE."""
        if not self.X:
            return
        res = minimize(self.logistic_loss, self.theta_hat, method="BFGS")
        self.theta_hat = res.x
        X = np.array(self.X)
        logits = X @ self.theta_hat
        probs = 1 / (1 + np.exp(-logits))
        W = np.diag(probs * (1 - probs))  # Diagonal weight matrix
        self.V = X.T @ W @ X + np.eye(self.d) * 1e-6  # Fisher Information Matrix

    def select_arm(self, arms):
        """Select an arm based on UCB criterion."""
        best_arm = None
        best_ucb = -np.inf

        for i, x_k in enumerate(arms):
            x_k = np.array(x_k)
            mean = x_k @ self.theta_hat
            variance = x_k @ np.linalg.inv(self.V) @ x_k
            ucb = mean + self.alpha * np.sqrt(variance)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = i

        return best_arm

    def update(self, x_k, reward):
        """Update the model with a new observation."""
        self.X.append(x_k)
        self.y.append(reward)
        self.fit()

# Example usage
np.random.seed(42)
d = 5  # Feature dimension
bandit = LogisticUCB(d)

# Generate some random arms (feature vectors)
num_arms = 10
arms = [np.random.randn(d) for _ in range(num_arms)]

# Simulate pulling arms
for _ in range(100):
    arm_idx = bandit.select_arm(arms)
    x_k = arms[arm_idx]
    true_theta = np.random.randn(d)  # True but unknown parameter
    reward = np.random.rand() < (1 / (1 + np.exp(-x_k @ true_theta)))  # Logistic reward
    bandit.update(x_k, reward)

print("Estimated theta:", bandit.theta_hat)
