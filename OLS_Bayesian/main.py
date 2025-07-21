
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



'''set random seed for reproducibility'''
random_state = 42
np.random.seed(random_state)
'''defeine hyperparameters'''
learning_rate = 0.001
lambda_val_lst = [0, 0.001, 0.01, 0.1, 1,2, 5] #list of lambda values
#lambda_val_lst = [0, 0.001, 0.01, 0.1, 1,2, 5,10, 50, 100] #list of lambda values
max_iterations = 100
theta_0_num = 50
alpha_val = 0.5
reduction = 'sum'

'''Generate sparse data'''
n_samples = 50
n_features = 100
output_features = 1
data_gen = DataGenerator(random_state = random_state)
X, y, _ = data_gen.get_linear_regression_data(n_samples=n_samples, n_features=n_features)



def inference(theta_0_array,input_x, X, Y, lambda_val, max_iterations= max_iterations, alpha_val= 0.5, learning_rate=0.01 ):
    """
Run gradient descent for multiple initial theta values sequentially

Args:
    theta_0_array: Shape (n_features, n_samples) - multiple initial theta vectors
"""
    f_out_lst = []
    theta_0_num = theta_0_array.shape[1]
    #print("shape of theta_0_array", theta_0_array.shape)
    for j in range(theta_0_num):
        theta_0 = theta_0_array[:, j]  # Select the j-th initial theta vector
        #print("Theta 0 value", theta_0_array)
        #print("Theta 0 shape: ", theta_0.shape)
        #input,X_matrix, y_vector, lambda_val,theta_0_array, max_iterations, alpha, eta
        gd = GradientDescent( input_x, 
                             X, Y,
                             theta_0, 
                             max_iterations, 
                             alpha_val, 
                             learning_rate, 
                             lambda_val,
                             reduction
                             )
        f_out_lst.append(gd.gradient_output())
        #print(gd.gradient_output())
    #check the variance of the output
    #print("f_out_lst: ", f_out_lst)
    variance = (np.var(f_out_lst, ddof=1))
    return variance

if __name__ == "__main__":
    ##generate the parameters W*
    theta_star = np.ones(n_features) 
    #Generate data
    init = InitParameter( dim = n_features, n_sample = theta_0_num, random_state = random_state) 
    theta_0_array = init.initialization()
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
    U = np.eye(n_features) - P
    ones = np.ones(n_features) # Create a vector of ones with the same dimension as d
    P_X = P @ ones
    U_X = U @ ones

     #may tune this hyperparameter
    var_P_lst = []
    var_U_lst = []

    var_P_C_lst = []
    var_U_C_lst = []


    for lambda_val in lambda_val_lst:


        var_P = inference(theta_0_array, P_X, X, y, lambda_val, max_iterations, alpha_val, learning_rate)
        var_P_lst.append(var_P)
        var_U = inference(theta_0_array, U_X, X, y, lambda_val, max_iterations, alpha_val, learning_rate)
        var_U_lst.append(var_U)

        '''
        config for closed-form solution. 
        As we have the lambda_val_list, we need to put the config inside here
        '''
        CLOSED_FORM_CONFIG = {
                        'X': X,           # np.ndarray of shape (n_samples, n_features)
                        'y': y,         # np.ndarray of shape (n_samples,)
                        'lambda_val':lambda_val,   # float (regularization parameter)
                        'theta_0_array': theta_0_array   # np.ndarray of shape (n_features,)
        }
        cfs  = ClosedFormSolver(CLOSED_FORM_CONFIG)
        var_P_C = cfs.mean_inference(P_X)[1]
        var_P_C_lst.append(var_P_C)
        var_U_C = cfs.mean_inference(U_X)[1]
        var_U_C_lst.append(var_U_C)

#plot two lines
plt.figure(figsize=(10, 6))
plt.plot(lambda_val_lst, var_P_lst, marker='o' ,linewidth=2, markersize=6, alpha = 0.5, label='Variance from projection subspace')
plt.plot(lambda_val_lst, var_P_C_lst, marker='^', linewidth=2, linestyle = '--', markersize=6, label='Variance from projection subspace (closed form)')
plt.plot(lambda_val_lst, var_U_lst, marker='s', linewidth=2,alpha =0.5, markersize=6, label='Variance from orthogonal subspace')
plt.plot(lambda_val_lst, var_U_C_lst, marker='d', linewidth=2, linestyle = '--', markersize=6, label='Variance from orthogonal subspace (closed form)')

# Customize the plot
plt.xlabel('value of lambda')
plt.ylabel('Variance')
plt.title(f"Variance vs lambda value (d = {n_features}, n = {n_samples}, initial_sample = {theta_0_num}, learning rate = {learning_rate}, alpha = {alpha_val})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f'plots/variance_ridge_lambda.png', dpi=300, bbox_inches='tight')
# Display the plot