import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000
t = np.linspace(0, 1, N, endpoint=False)
f0 = 1  # fundamental frequency

# Square wave target
y = np.sign(np.sin(2 * np.pi * f0 * t))  # Square wave

# Build design matrix with odd harmonics only
harmonics = [1, 3, 5, 7, 9]
X = np.column_stack([np.sin(2 * np.pi * k * f0 * t) for k in harmonics])
print(X.shape , "shape of X")

# Solve regression
theta = np.linalg.lstsq(X, y, rcond=None)[0]
print(theta.shape, "shape of theta")

# Reconstruct signal
y_hat = X @ theta
print(y_hat.shape, "shape of y_hat")

# Plot
plt.plot(t, y, label='Square wave (target)', linestyle='--')
plt.plot(t, y_hat, label='Fourier regression')
plt.legend()
plt.title('Fourier Series Approximation of a Square Wave')
plt.show()
