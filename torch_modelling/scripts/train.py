'''import libraries'''

# Add src directory to Python path for imports
# This runs automatically for all tests in this directory
import sys
import os
from pathlib import Path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from models.linear_model import LinearModel
from models.np_init_parameter import InitParameter
from data.dataset import LinearDataset
from data.data_generator import DataGenerator
from training.trainer import Trainer
from models.closed_form_solver import ClosedFormSolver
from torch.utils.data import DataLoader
import numpy as np



'''set random seed for reproducibility'''
random_state = 42
np.random.seed(random_state)
'''defeine hyperparameters'''
learning_rate = 0.001
lambda_val_lst = [0, 0.001, 0.01, 0.1, 1,2, 5, 10] #list of lambda values
#lambda_val_lst = [0, 0.001, 0.01, 0.1, 1,2, 5,10, 50, 100] #list of lambda values
max_iterations = 100
theta_0_num = 50
alpha_val = 0.5

'''Generate sparse data'''
n_samples = 50
n_features = 100
output_features = 1
data_gen = DataGenerator(random_state = random_state)
X, y, _ = data_gen.get_linear_regression_data(n_samples=n_samples, n_features=n_features)
dataset = LinearDataset(X, y)

'''this is for each lambda value. I'm going to put this into a function? no let me put it into a for loop'''
var_P_X_lst = []
var_U_X_lst = []
P_X_lst = []
U_X_lst = []
P_C_lst = []
U_C_lst = []
for lambda_val in lambda_val_lst:
    '''theta_0 generation'''
    init_param = InitParameter(dim = n_features, n_samples = theta_0_num, random_state = random_state)
    theta_0_array = init_param.initialization()
    
    #initial the P_X and U_X stack
   
    for j in range (theta_0_num):
        theta_0 = theta_0_array[:, j]
        '''modelling'''
        model = LinearModel(input_channels=n_features, output_channels=output_features, theta_0=theta_0)  # Example dimensions

        gd_config = {
            "model" : model,
            "learning_rate" : learning_rate,
            "lambda_val" :lambda_val,
            "loss_fn" : nn.MSELoss(reduction= 'sum')
        }

        trainer = Trainer(gd_config)
        #I should input dataset instead of X and y here
        trainer.train(dataset.X, dataset.y, epochs = max_iterations)


        #create x_p and x_u as ols_bayesian
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

        pred_P_X = trainer.iterative_mean(P_X, max_iterations, alpha_val)
        pred_U_X = trainer.iterative_mean(U_X, max_iterations, alpha_val)
        #print(pred_P_X)
        #print(pred_U_X)
        P_X_lst.append(pred_P_X)
        U_X_lst.append(pred_U_X)
    '''closed form solutions'''
    CLOSED_FORM_CONFIG = {
                    'X': dataset.X,           # np.ndarray of shape (n_samples, n_features)
                    'y': dataset.y,         # np.ndarray of shape (n_samples,)
                    'lambda_val':lambda_val,   # float (regularization parameter)
                    'theta_0_array': theta_0_array   # np.ndarray of shape (n_features,)
    }
    cfs = ClosedFormSolver(CLOSED_FORM_CONFIG)
    P_C = cfs.mean_inference(P_X)[1]
    U_C = cfs.mean_inference(U_X)[1]
    P_C_lst.append(P_C)
    U_C_lst.append(U_C)

    P_X_stack = torch.stack(P_X_lst, dim=0)
    U_X_stack = torch.stack(U_X_lst, dim=0)
    var_P_X = torch.var(P_X_stack, dim=0)
    var_U_X = torch.var(U_X_stack, dim=0)
    var_P_X_lst.append(var_P_X)
    var_U_X_lst.append(var_U_X)


#plot two lines
plt.figure(figsize=(10, 6))
plt.plot(lambda_val_lst, var_P_X_lst, marker='o' ,linewidth=2, markersize=6, alpha = 0.5, label='Variance from projection subspace')
plt.plot(lambda_val_lst, P_C_lst, marker='^', linewidth=2, linestyle = '--', markersize=6, label='Variance from projection subspace (closed form)')
plt.plot(lambda_val_lst, var_U_X_lst, marker='s', linewidth=2,alpha =0.5, markersize=6, label='Variance from orthogonal subspace')
plt.plot(lambda_val_lst, U_C_lst, marker='d', linewidth=2, linestyle = '--', markersize=6, label='Variance from orthogonal subspace (closed form)')

# Customize the plot
plt.xlabel('value of lambda')
plt.ylabel('Variance')
plt.title(f"Variance vs lambda value (d = {n_features}, n = {n_samples}, initial_sample = {theta_0_num}, learning rate = {learning_rate}, alpha = {alpha_val})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("plots/pytorch_plot/", exist_ok=True)
plt.savefig(f'plots/pytorch_plot/variance_ridge_lambda.png', dpi=300, bbox_inches='tight')
# Display the plot
