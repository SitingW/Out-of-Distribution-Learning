import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math

# Parameters
delta = 0.05
theta_star = np.array([6, 8])
S = np.linalg.norm(theta_star, 2) + 1
Lt = lambda t: (1 + S / 2) * (t - 1)

n = 4000
X = np.random.randn(n, 2)
mu = lambda x: 1 / (1 + np.exp(-x))
r = np.random.binomial(1, mu(X @ theta_star))

# Likelihood function
likelihood = lambda theta: cp.sum(cp.multiply(r, cp.logistic(-X @ theta)) + cp.multiply((1 - r), cp.logistic(X @ theta)))

# Solve for theta_hat
theta = cp.Variable(2)
obj = likelihood(theta)
constraints = [cp.norm(theta, 2) <= S]
objective = cp.Minimize(obj)
problem = cp.Problem(objective, constraints)
problem.solve()
theta_hat = theta.value

# UCB optimization for theta2
theta2 = cp.Variable(2)
arm = np.array([3, 4]) / 5

obj = cp.sum(cp.multiply(theta2, arm))
constraints = [cp.norm(theta2, 2) <= S,
               likelihood(theta2) - likelihood(theta_hat) <= cp.log(1 / delta) + 2 * cp.log(max(math.e, math.e * S * Lt(n)))]

objective = cp.Maximize(obj)

problem = cp.Problem(objective, constraints)
problem.solve(verbose=True)

# Define the grid for theta2 values
theta1_range = np.linspace(-10, 10, 100)
theta2_range = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(theta1_range, theta2_range)

confidence_set = np.zeros_like(X)

# Evaluate each (theta1, theta2) pair
for i in range(X.shape[0]):
    for j in range(Y.shape[0]):
        theta_value = np.array([X[j], Y[j]])
        constraints[1] = likelihood(theta_value) - likelihood(theta_hat) <= np.log(1 / delta) + 2 * np.log(max(math.e, math.e * S * Lt(n)))
        problem = cp.Problem(cp.Maximize(obj), constraints)
        problem.solve()
        confidence_set[i, j] = problem.status == cp.OPTIMAL  # Record if optimal solution is found

# Plot heatmap of the confidence set
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, confidence_set, cmap="Blues", alpha=0.7)
plt.colorbar(label="Confidence Set")
plt.xlabel("Theta1")
plt.ylabel("Theta2")
plt.title("Confidence Set Heatmap")
plt.show()
