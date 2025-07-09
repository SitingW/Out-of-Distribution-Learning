
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import DataGenerator
from gradient_descent import GradientDescent
from init_parameter import InitParameter
from closed_form_solver import ClosedFormSolver

#set random seeds
np.random.seed(42)

#define the input output space
d = 100
n = 50
o = 1 #output dimension


def inference(theta_0_array,input_x, X, Y, lambda_val, max_iterations, alpha, learning_rate):
    """
Run gradient descent for multiple initial theta values sequentially

Args:
    theta_0_array: Shape (n_features, n_samples) - multiple initial theta vectors
"""
    f_out_lst = []
    n_samples = theta_0_array.shape[1]
    
    for j in range(n_samples):
        theta_0 = theta_0_array[:, j]  # Select the j-th initial theta vector
        #print(theta_0_array)
        #input,X_matrix, y_vector, lambda_val,theta_0_array, max_iterations, alpha, eta
        gd = GradientDescent( input_x , X, Y,theta_0, max_iterations, alpha, learning_rate,  lambda_val)
        f_out_lst.append(gd.gradient_output())
        #print(gd.gradient_output())
    variance = (np.var(f_out_lst, ddof=1))
    return variance

if __name__ == "__main__":
    ##generate the parameters W*
    theta_star = np.ones(d) 
    #Generate data
    linear_gen = DataGenerator(random_state = 42)
    X, Y, theta_star = linear_gen.linear_regression_data(n_samples=n, n_features= d) 

    lambda_val = 0
    initial_ite_lst = [30, 50 , 100, 200] 
    learning_rate=0.001
    max_iterations = n
    alpha = 0.5
    
    
   
    #input_x = np.eye(d)[30] #random generation
    #init will give n 
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

     #may tune this hyperparameter
    var_P_lst = []
    var_U_lst = []
    for initial_ite in initial_ite_lst:
        init = InitParameter( dim = d, n_sample = initial_ite) 
        theta_0_array = init.initialization()

        var_P = inference(theta_0_array, P_X, X, Y, lambda_val, max_iterations, alpha, learning_rate)
        var_P_lst.append(var_P)
        var_U = inference(theta_0_array, U_X, X, Y, lambda_val, max_iterations, alpha, learning_rate)
        var_U_lst.append(var_U)

#plot two lines
plt.figure(figsize=(10, 6))
plt.plot(initial_ite_lst, var_P_lst, marker='o', linewidth=2, markersize=6, label='Variance from projection subspace')
plt.plot(initial_ite_lst, var_U_lst, marker='s', linewidth=2, markersize=6, label='Variance from orthogonal subspace')

# Customize the plot
plt.xlabel('value of lambda')
plt.ylabel('Variance')
plt.title("Variance vs lambda value (d=100, n=50, initial_sample ="+ str(initial_ite)+ "), lambda value = "+ str(lambda_val)+ ")")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f'plots/variance_initial_val.png', dpi=300, bbox_inches='tight')
# Display the plot