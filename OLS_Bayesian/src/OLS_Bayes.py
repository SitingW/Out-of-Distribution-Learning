#import library
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
#import math
from scipy.linalg import orth
import os
#set random seeds
np.random.seed(42)

#define the input output space
d = 100
n = 50
o = 1 #output dimension
##generate the parameters W*
theta_star = np.ones(d)


class DataGenerator:
    def __init__(self, random_state = None):
        if random_state:
            # Use proper random state handling with Generator
            self.rng = np.random.Generator(np.random.PCG64(random_state))

    def linear_regression_data (self, n_samples, n_features):
        X = np.random.randn(n_samples, n_features)
        true_theta = np.random.randn(n_features)
        y = X @ true_theta
        return X, y, true_theta

class GradientDescent:
    """
    Args:
        lambda_val: regularization weight
        alpha: iterative mean weight
        eta: (removed from the class, an unnecessary hyperparameter)
    """
    def __init__(self,input,X_matrix, y_vector, lambda_val,theta_0, max_iterations, alpha):
        self.lambda_val = lambda_val
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.theta_history = [theta_0]
        self.f_bar_lst = [input @ theta_0]
        self.input = input
        self.X_train = X_matrix
        self.y_train = y_vector
        

    def predict(self, x, theta):
        print(x.shape, theta.shape)
        return x @ theta
    
    
    #change this part into computing bunch gradients instead of one
    def gradient_compute(self, x, y):

        theta_new = self.theta_history[-1] - self.lambda_val * np.mean(x.T @ (self.predict(x , self.theta_history[-1]) - y) )
        self.theta_history.append(theta_new)
        return theta_new
    
    def iterative_avg(self):
        f_bar = self.alpha * self.predict(self.input,self.theta_history[-1] ) + ( 1- self.alpha) * self.f_bar_lst[-1]
        self.f_bar_lst.append(f_bar)

    def gradient_output(self):
        for i in range (self. max_iterations):
            self.gradient_compute(self.X_train, self.y_train)
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


# class Inference():
#     def __init__(self):
#         self.f_out_lst = []

#     def inference(self, initial_ite):
#         f_out_lst = []
#         for j in range(initial_ite):
#             theta_0 = init.initialization()
#             #print(theta_0)
#             #input,X_matrix, y_vector, lambda_val,theta_0, max_iterations, alpha, eta
#             gd = GradientDescent( input_x , X, Y, lambda_val,theta_0, max_iterations= n, alpha= 0.5, eta=0.01)
#             f_out_lst.append(gd.gradient_output()) 


"""
Why I'm doing this? For creating a loop for multiple input x for further comparison
"""
def inference(initial_ite,input_x, X, Y, lambda_val, max_iterations= n, alpha= 0.5, eta=0.01 ):
        f_out_lst = []
        for j in range(initial_ite):
            theta_0 = init.initialization()
            #print(theta_0)
            #input,X_matrix, y_vector, lambda_val,theta_0, max_iterations, alpha, eta
            gd = GradientDescent( input_x , X, Y, lambda_val,theta_0, max_iterations, alpha, eta)
            f_out_lst.append(gd.gradient_output())
            #print(gd.gradient_output())
        variance = (np.var(f_out_lst, ddof=1))
        return variance

if __name__ == "__main__":
    #Generate data
    linear_gen = DataGenerator(random_state = 42)
    X, Y, theta_star = linear_gen.linear_regression_data(n_samples=n, n_features= d) 

    lambda_val = 1
  
    init = InitParameter( dim = d, lambda_val = lambda_val) 
    #input_x = np.eye(d)[30] #random generation



    #seperate the subspaces
    #I_d - X(X^TX)^{-1}X^T
    #use scipy svd get two subspaces
    def projection_matrix_qr(X):
        Q, R = np.linalg.qr(X)
        return Q @ Q.T
    P = projection_matrix_qr (X.T)
    #print("Projection matrix P shape: ", P.shape)
    U = np.eye(d) - P
    ones = np.ones(d) # Create a vector of ones with the same dimension as d
    P_X = P @ ones
    U_X = U @ ones

    initial_ite_lst = [5, 10 , 30 , 50 , 75, 100 , 200, 500] #may tune this hyperparameter
    var_P_lst = []
    var_U_lst = []
    for initial_ite in initial_ite_lst:

        var_P = inference(initial_ite,P_X, X, Y, lambda_val, max_iterations= n, alpha= 0.5, eta=0.01)
        var_P_lst.append(var_P)
        var_U = inference(initial_ite,U_X, X, Y, lambda_val, max_iterations= n, alpha= 0.5, eta=0.01)
        var_U_lst.append(var_U)

#plot two lines
plt.figure(figsize=(10, 6))
plt.plot(initial_ite_lst, var_P_lst, marker='o', linewidth=2, markersize=6, label='Variance from projection subspace')
plt.plot(initial_ite_lst, var_U_lst, marker='s', linewidth=2, markersize=6, label='Variance from orthogonal subspace')

# Customize the plot
plt.xlabel('# of initializations')
plt.ylabel('Variance')
plt.title('Variance vs Number of initializations (d=100, n=50)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f'plots/variance_init_plot.png', dpi=300, bbox_inches='tight')
# Display the plot