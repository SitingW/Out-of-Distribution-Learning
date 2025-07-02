#inport library
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math

#set random seeds
np.random.seed(42)

#define the input outpue space
d = 100
n = 50
o = 1 #output dimension
##generate the parameters W*
theta_star = np.ones(d)


class DataGenerator:
    def __init__(self, random_state = None):
        if random_state:
            np.random.seed(random_state)

    def linear_regression_data (self, n_samples = n, n_features = d):
        X = np.random.randn(n_samples, n_features)
        true_theta = np.random.randn(n_features)
        y = X @ true_theta
        return X, y, true_theta

class GradientDescent:
    def __init__(self, lambda_val, max_iterations, alpha, eta):
        self.lambda_val = lambda_val
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.eta =eta
        self.theta_history = []
        self.f_bar_lst = []

    def predict(self, x, theta):
        return x @ theta
    
    def gradient_compute(self, x, y):
        theta_new = self.theta_history[-1] - self.eta * x * (self.predict(x ,self.theta_history[-1]) - y)
        self.theta_history.append(theta_new)
        return theta_new
    
    def iterative_avg(self, x):
        f_bar = self.alpha * self.predict(x,self.theta_history[-1] ) + ( 1- self.alpha) * self.f_bar[-1]
        self.f_bar_lst.append(f_bar)
    

# Define the Initialization function

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







if __name__ == "__main__":
    lambda_val = 1
    init = InitParameter( dim = d, lambda_val = lambda_val)
    theta_0 = init.initialization()
    print(theta_0)