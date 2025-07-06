from dependencies import np

class InitParameter:
    def __init__(self, dim,  lambda_val= 1, cov = None):
        if cov is None:
            cov = np.eye(dim)
        self.dim = dim
        self.cov = cov
        self.lambda_val = lambda_val
    
    def initialization(self):
        mean = np.zeros(self.dim)
        cov = 1/ self.lambda_val * self.cov
        return np.random.multivariate_normal (mean, cov)
        