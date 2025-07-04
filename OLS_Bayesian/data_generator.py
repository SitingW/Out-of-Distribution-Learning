class DataGenerator:
    def __init__(self, random_state = None):
        if random_state:
            np.random.seed(random_state)

    def linear_regression_data (self, n_samples = n, n_features = d):
        X = np.random.randn(n_samples, n_features)
        true_theta = np.random.randn(n_features)
        y = X @ true_theta
        return X, y, true_theta