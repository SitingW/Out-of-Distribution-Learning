
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


def inference(theta_0_array,input_x, X, Y, lambda_val, max_iterations, alpha, learning_rate ):
    """
Run gradient descent for multiple initial theta values sequentially

Args:
    theta_0_array: Shape (n_features, n_samples) - multiple initial theta vectors
"""
    f_out_lst = []
    n_samples = theta_0_array.shape[1]
    #print("Theta 0 array shape: ", theta_0_array.shape)
    #print("Number of initial theta vectors: ", n_samples)
    
    for j in range(n_samples):
        theta_0 = theta_0_array[:, j]  # Select the j-th initial theta vector
        #print("Theta 0 value", theta_0_array)
        #print("Theta 0 shape: ", theta_0.shape)
        #input,X_matrix, y_vector, lambda_val,theta_0_array, max_iterations, alpha, eta
        gd = GradientDescent( input_x , X, Y,theta_0, max_iterations, alpha, learning_rate,  lambda_val)
        f_out_lst.append(gd.gradient_output())
        #print(gd.gradient_output())
    #check the variance of the output
    #print("f_out_lst: ", f_out_lst)
    
    q10 = np.percentile(f_out_lst, 10)
    median = np.percentile(f_out_lst, 50)  # or np.median(data)
    q90 = np.percentile(f_out_lst, 90)
    
    return q10, median, q90


if __name__ == "__main__":
    ##generate the parameters W*
    theta_star = np.ones(d) 
    #Generate data
    linear_gen = DataGenerator(random_state = 48)
    X, Y, theta_star = linear_gen.linear_regression_data(n_samples=n, n_features= d) 
    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)

    lambda_val_lst = [0, 0.001, 0.01, 0.1, 1,2, 5]
    initial_ite = 50 
    max_iterations = 500
    learning_rate=0.001
    alpha = 0.5
  
    init = InitParameter( dim = d, n_sample = initial_ite) 
    theta_0_array = init.initialization()
    #input_x = np.eye(d)[30] #random generation
    #init will give n 
    #seperate the subspaces
    #I_d - X(X^TX)^{-1}X^T
    #use scipy svd get two subspaces
    def projection_matrix_qr(X):
        Q, R = np.linalg.qr(X, mode = "reduced")
        return Q @ Q.T
    P = projection_matrix_qr (X.T)
    #print("Projection matrix P shape: ", P.shape)
    U = np.eye(d) - P
    ones = np.ones(d) # Create a vector of ones with the same dimension as d
    P_X = P @ ones
    U_X = U @ ones

     #may tune this hyperparameter
    q10_P_lst = []
    q50_P_lst = []
    q90_P_lst = []
    q10_U_lst = []
    q50_U_lst = []
    q90_U_lst = []

    q10_P_C_lst = []
    q50_P_C_lst = []
    q90_P_C_lst = []
    q10_U_C_lst = []
    q50_U_C_lst = []
    q90_U_C_lst = []
    
    for lambda_val in lambda_val_lst:
        #closed for solutions
        cfs = ClosedFormSolver(X, Y, theta_0_array, lambda_val)
        _, _, q10_P_C, q50_P_C, q90_P_C = cfs.inference(P_X)
        q10_P_C_lst.append(q10_P_C)
        q50_P_C_lst.append(q50_P_C)
        q90_P_C_lst.append(q90_P_C)
        _, _, q10_U_C, q50_U_C, q90_U_C = cfs.inference(U_X)
        q10_U_C_lst.append(q10_U_C)
        q50_U_C_lst.append(q50_U_C)
        q90_U_C_lst.append(q90_U_C)

        q10_P, q50_P, q90_P = inference(theta_0_array, P_X, X, Y, lambda_val, max_iterations, alpha, learning_rate)
        q10_P_lst.append(q10_P)
        q50_P_lst.append(q50_P)
        q90_P_lst.append(q90_P)
        q10_U, q50_U, q90_U = inference(theta_0_array, U_X, X, Y, lambda_val, max_iterations, alpha, learning_rate)
        q10_U_lst.append(q10_U)
        q50_U_lst.append(q50_U)
        q90_U_lst.append(q90_U)

theta_0_q10 = np.percentile(theta_0_array, 10)
theta_0_median = np.percentile(theta_0_array, 50)
theta_0_q90 = np.percentile(theta_0_array, 90)

# Create the plot
plt.figure(figsize=(10, 6))


# Plot P (projection subspace) - first color
plt.plot(lambda_val_lst, q50_P_lst, marker='o', linewidth=2, markersize=6, 
         label='Median from projection subspace', color='blue')
plt.fill_between(lambda_val_lst, q10_P_lst, q90_P_lst, alpha=0.15, color='blue',
                 label='10%-90% quantile interval (projection)')

# Plot U (orthogonal subspace) - second color  
plt.plot(lambda_val_lst, q50_U_lst, marker='s', linewidth=2, markersize=6,
         label='Median from orthogonal subspace', color='orange')
plt.fill_between(lambda_val_lst, q10_U_lst, q90_U_lst, alpha=0.15, color='orange',
                 label='10%-90% quantile interval (orthogonal)')

# Add horizontal quantile intervals for theta_0
plt.axhline(y=theta_0_q10, color='red', linestyle='--', alpha=0.7, linewidth=1)
plt.axhline(y=theta_0_q90, color='red', linestyle='--', alpha=0.7, linewidth=1)

# Fill between the theta_0 quantiles (optional - creates a horizontal band)
plt.fill_between(lambda_val_lst, theta_0_q10, theta_0_q90, alpha=0.1, color='red',
                 label='Theta_0 10%-90% interval')

# Customize the plot
plt.xlabel('value of lambda')
plt.ylabel('Response')
plt.title(f"Quantiles vs lambda value (d=100, n=50, initial_sample ={initial_ite}, max_iterations = {max_iterations}, learning rate = {learning_rate}, alpha = {alpha})")
plt.legend()
plt.grid(True, alpha=0.15)
plt.tight_layout()

# Save the plot
os.makedirs("plots", exist_ok=True)
plt.savefig(f'plots/quantiles_ridge_lambda.png', dpi=300, bbox_inches='tight')
plt.close()

# Gradient Descent results for P
plt.plot(lambda_val_lst, q50_P_lst, marker='o', linewidth=2, markersize=6, 
         label='Gradient Descent (median)', color='blue')
plt.fill_between(lambda_val_lst, q10_P_lst, q90_P_lst, alpha=0.15, color='blue',
                 label='Gradient Descent (10%-90% interval)')

# Closed Form results for P
plt.plot(lambda_val_lst, q50_P_C_lst, marker='s', linewidth=2, markersize=6,
         label='Closed Form (median)', color='green')
plt.fill_between(lambda_val_lst, q10_P_C_lst, q90_P_C_lst, alpha=0.15, color='green',
                 label='Closed Form (10%-90% interval)')

# Add horizontal line for theta_0 reference
plt.axhline(y=theta_0_median, color='red', linestyle='-', linewidth=2, 
            label='Theta_0 median')
plt.axhline(y=theta_0_q10, color='red', linestyle='--', alpha=0.5, linewidth=1)
plt.axhline(y=theta_0_q90, color='red', linestyle='--', alpha=0.5, linewidth=1)
plt.fill_between(lambda_val_lst, theta_0_q10, theta_0_q90, alpha=0.1, color='red',
                 label='Theta_0 (10%-90% interval)')

plt.xlabel('Lambda value')
plt.ylabel('Response')
plt.title(f"P (Known) Subspace: Gradient Descent vs Closed Form\n(d=100, n=50, initial_sample = {initial_ite}, max_iterations = {max_iterations}, learning rate = {learning_rate}, alpha = {alpha})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/P_known_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


# Plot 2: U (Unknown) subspace comparison
plt.figure(figsize=(12, 6))

# Gradient Descent results for U
plt.plot(lambda_val_lst, q50_U_lst, marker='o', linewidth=2, markersize=6, 
         label='Gradient Descent (median)', color='blue')
plt.fill_between(lambda_val_lst, q10_U_lst, q90_U_lst, alpha=0.15, color='blue',
                 label='Gradient Descent (10%-90% interval)')

# Closed Form results for U
plt.plot(lambda_val_lst, q50_U_C_lst, marker='s', linewidth=2, markersize=6,
         label='Closed Form (median)', color='green')
plt.fill_between(lambda_val_lst, q10_U_C_lst, q90_U_C_lst, alpha=0.15, color='green',
                 label='Closed Form (10%-90% interval)')

# Add horizontal line for theta_0 reference
plt.axhline(y=theta_0_median, color='red', linestyle='-', linewidth=2, 
            label='Theta_0 median')
plt.axhline(y=theta_0_q10, color='red', linestyle='--', alpha=0.5, linewidth=1)
plt.axhline(y=theta_0_q90, color='red', linestyle='--', alpha=0.5, linewidth=1)
plt.fill_between(lambda_val_lst, theta_0_q10, theta_0_q90, alpha=0.1, color='red',
                 label='Theta_0 (10%-90% interval)')

plt.xlabel('Lambda value')
plt.ylabel('Response')
plt.title(f"U (Unknown) Subspace: Gradient Descent vs Closed Form\n(d=100, n=50, initial_sample = {initial_ite}, max_iterations = {max_iterations}, learning rate = {learning_rate}, alpha = {alpha})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/U_unknown_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots saved as:")
print("- plots/P_known_comparison.png")
print("- plots/U_unknown_comparison.png")