from __init__ import __init__

from data_generator import DataGenerator
from gradient_descent import GradientDescent
from init_parameter import InitParameter

if __name__ == "__main__":
    # Generate data
    linear_gen = DataGenerator(random_state=42)
    n = 10  # Number of samples
    d = 100  # Number of features
    X, y, true_theta = linear_gen.linear_regression_data(n_samples=n, n_features=d)
    lambda_val = 1
    init = InitParameter(dim=d, lambda_val=lambda_val)

    def projection_matrix_qr(X):
        Q, R = np.linalg.qr(X)
        return Q @ Q.T
    P = projection_matrix_qr (X.T)
    U = np.eye(d) - P
    ones = np.ones(d)
    P_X = P @ ones
    U_X = P @ ones