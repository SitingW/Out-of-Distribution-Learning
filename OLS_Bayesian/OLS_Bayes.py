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
    def __init__(self,input,X_matrix, y_vector, lambda_val,theta_0, max_iterations, alpha, eta):
        self.lambda_val = lambda_val
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.eta =eta
        self.theta_history = [theta_0]
        self.f_bar_lst = [input @ theta_0]
        self.input = input
        self.X_train = X_matrix
        self.y_train = y_vector
        

    def predict(self, x, theta):
        return x @ theta
    
    
    
    def gradient_compute(self, x, y):
        theta_new = self.theta_history[-1] - self.eta * x * (self.predict(x ,self.theta_history[-1]) - y)
        self.theta_history.append(theta_new)
        return theta_new
    
    def iterative_avg(self):
        f_bar = self.alpha * self.predict(self.input,self.theta_history[-1] ) + ( 1- self.alpha) * self.f_bar_lst[-1]
        self.f_bar_lst.append(f_bar)

    def gradient_output(self):
        for i in range (self. max_iterations):
            self.gradient_compute(self.X_train[i], self.y_train[i])
            self.iterative_avg()
        return self.f_bar_lst[-1]
    

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


class Inference():
    def __init__(self):
        self.f_out_lst = []

    def inference(self):
       f_out_lst = []
    for j in range(initial_ite):
        theta_0 = init.initialization()
        #print(theta_0)
        #input,X_matrix, y_vector, lambda_val,theta_0, max_iterations, alpha, eta
        gd = GradientDescent( input_x , X, Y, lambda_val,theta_0, max_iterations= n, alpha= 0.5, eta=0.01)
        f_out_lst.append(gd.gradient_output()) 




if __name__ == "__main__":
    #Generate data
    linear_gen = DataGenerator(random_state = 42)
    X, Y, theta_star = linear_gen.linear_regression_data(n_samples=n, n_features= d) 

    lambda_val = 1
    initial_ite = 30
    init = InitParameter( dim = d, lambda_val = lambda_val) 
    input_x = np.eye(d)[30] #random generation

    f_out_lst = []
    for j in range(initial_ite):
        theta_0 = init.initialization()
        #print(theta_0)
        #input,X_matrix, y_vector, lambda_val,theta_0, max_iterations, alpha, eta
        gd = GradientDescent( input_x , X, Y, lambda_val,theta_0, max_iterations= n, alpha= 0.5, eta=0.01)
        f_out_lst.append(gd.gradient_output())
        #print(gd.gradient_output())

    print(np.var(f_out_lst, ddof=1))