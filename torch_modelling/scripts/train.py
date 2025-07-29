'''import libraries'''

# Add src directory to Python path for imports
# This runs automatically for all tests in this directory
import sys
import os
from pathlib import Path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.models.linear_model import LinearModel
from src.models.np_init_parameter import InitParameter
from src.data.dataset import LinearDataset
from src.data.data_generator import DataGenerator
from src.training.trainer import Trainer
from src.models.closed_form_solver import ClosedFormSolver
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any




#random_state = 42
def set_random_seeds(random_state: int) -> None:
    """set random seed for reproducibility"""
    np.random.seed(random_state)
    torch.manual_seed(random_state)


"""define data generation"""
def generate_data(n_samples: int, n_features: int, random_state: int)-> Tuple[LinearDataset, np.ndarray, np.ndarray]:
    data_gen = DataGenerator(random_state = random_state)
    X, y, _ = data_gen.get_linear_regression_data(n_samples=n_samples, n_features=n_features)
    dataset = LinearDataset(X, y)
    return dataset, X, y

def projection_matrix_qr(X: np.ndarray) -> np.ndarray:
    """define the projection matrix"""
    Q, R = np.linalg.qr(X)
    return Q @ Q.T

def get_projection_matrices (X: np.ndarray, n_features: int) -> Tuple[np.ndarray , np.ndarray]:
    """
    Create projection matrix and orthogonal matrix.
    Function is borrowed from numpy codebase.
    """
    #Changed the name from matrices from P, U into proj_matrix and orth_matrix for clearity.
    #Some issue might be that I'm reusing the name of proj_matrix and orth_matrix, which might leads to trouble in the future.
    proj_matrix = projection_matrix_qr (X.T)
    #print("Projection matrix P shape: ", P.shape)
    orth_matrix = np.eye(n_features) - proj_matrix
    ones = np.ones(n_features) # Create a vector of ones with the same dimension as d
    proj_matrix = proj_matrix @ ones
    orth_matrix = orth_matrix @ ones
    return proj_matrix, orth_matrix

#define single run training for reuse
def train_single_model(n_features: int, output_features: int, theta_0: float, learning_rate: float, lambda_val: float, dataset: LinearDataset, max_iterations: int) -> Trainer:
    """
    Train a single ridge regression model w.r.t. a special initialization parameter (theta_0), learning rate (learning_rate), and a L2 regularization term (lambda_val).
    """
    model = LinearModel(input_channels=n_features, output_channels=output_features, theta_0=theta_0)  #Example dimensions

    gd_config = {
        "model" : model,
        "learning_rate" : learning_rate,
        "lambda_val" :lambda_val,
        "loss_fn" : nn.MSELoss(reduction= 'sum')
    }

    trainer = Trainer(gd_config)
    #I should input dataset instead of X and y here
    trainer.train(dataset.X, dataset.y, epochs = max_iterations)
    return trainer

#define compute variance w.r.t. each lambda value
#eventually I want to compute this w.r.t. different hyperparameters
#maybe I gonna change the nem into compute the variance for a loop??
def compute_variance_wrt_theta_init(
        proj_matrix: np.ndarray,
        orth_matrix: np.ndarray,
        n_features: int, 
        output_features: int,
        theta_0_num: int, 
        random_state: int, 
        lambda_val: float,
        dataset: LinearDataset, 
        max_iterations: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute both the iterative and closed-form variance of precition w.r.t. one set of hyperparameters.
    Call the list of initial parameters, and the set of hyperparameters.
    trainer.iterative_mean will make sure convert every np ndarray into torch tensor.
    for each initial parameter, call the trainer to train the model. With the given input (pred_matrix, otrh_matrix), compute the prediction values.
    """
    init_param = InitParameter(dim = n_features, n_samples = theta_0_num, random_state = random_state)
    theta_0_array = init_param.initialization()
    """
    Change name from P_X_lst into proj_matrix_lst.
    Change name from U_X_lst into orth_matrix_lst.
    """
    proj_matrix_lst = []
    orth_matrix_lst = []
    #initial the P_X and U_X stack
   
    for j in range (theta_0_num):
        theta_0 = theta_0_array[:, j]

        trainer = train_single_model(n_features, output_features, theta_0, learning_rate, lambda_val, dataset, max_iterations)

        

        """
        change name from pred_P_X into pred_proj_matrix for clarity.
        change name from pred_U_X into pred_proj_matrix for clarity.
        """
        pred_proj_matrix = trainer.iterative_mean(proj_matrix, max_iterations, alpha_val)
        pred_orth_matrix = trainer.iterative_mean(orth_matrix, max_iterations, alpha_val)
        #print(pred_P_X)
        #print(pred_U_X)
        
        proj_matrix_lst.append(pred_proj_matrix)
        orth_matrix_lst.append(pred_orth_matrix)
    '''closed form solutions'''
    CLOSED_FORM_CONFIG = {
                    'X': dataset.X,           # np.ndarray of shape (n_samples, n_features)
                    'y': dataset.y,         # np.ndarray of shape (n_samples,)
                    'lambda_val':lambda_val,   # float (regularization parameter)
                    'theta_0_array': theta_0_array   # np.ndarray of shape (n_features,)
    }
    cfs = ClosedFormSolver(CLOSED_FORM_CONFIG)
    P_C = cfs.mean_inference(proj_matrix)[1]
    U_C = cfs.mean_inference(orth_matrix)[1]
    

    """Change name from P_X_stack & U_X_stack into proj_matrix_stack & orth_matrix_stack"""
    proj_matrix_stack = torch.stack(proj_matrix_lst, dim=0)
    orth_matrix_stack = torch.stack(orth_matrix_lst, dim=0)
    """Change name from var_P_X & var_U_X into var_proj_matrix & var_orth_matrix"""
    var_proj_matrix = torch.var(proj_matrix_stack, dim=0)
    var_orth_matrix = torch.var(orth_matrix_stack, dim=0)

    return var_proj_matrix, var_orth_matrix, P_C, U_C




"""define random state"""
random_state = 42
'''defeine hyperparameters'''
learning_rate = 0.001
lambda_val_lst = [0, 0.001, 0.01, 0.1, 1,2, 5, 10] #list of lambda values
#lambda_val_lst = [0, 0.001, 0.01, 0.1, 1,2, 5,10, 50, 100] #list of lambda values
max_iterations = 100
theta_0_num = 50
alpha_val = 0.5
n_samples = 50
n_features = 100
output_features = 1

"""Set-up random seeds"""
set_random_seeds(random_state)

"""Generate sparse data"""
data_gen = DataGenerator(random_state = random_state)
X, y, _ = data_gen.get_linear_regression_data(n_samples=n_samples, n_features=n_features)
dataset = LinearDataset(X, y)

'''this is for each lambda value. I'm going to put this into a function? no let me put it into a for loop'''
"""Change name from var_P_X_lst into var_proj_matrix_lst for clearity."""
var_proj_matrix_lst = []
var_orth_matrix_lst = []
P_C_lst = []
U_C_lst = []


"""run get_projection_matrices(X, n_fratures) to get the projection matrices"""
proj_matrix, orth_matrix = get_projection_matrices(X, n_features)

for lambda_val in lambda_val_lst:
    var_proj_matrix, var_orth_matrix, P_C, U_C = compute_variance_wrt_theta_init(
        proj_matrix, orth_matrix,
        n_features, output_features,
        theta_0_num, random_state,lambda_val, dataset, max_iterations
    )
    P_C_lst.append(P_C)
    U_C_lst.append(U_C)

    """Change name from var_P_X & var_U_X into var_proj_matrix & var_orth_matrix"""

    var_proj_matrix_lst.append(var_proj_matrix)
    var_orth_matrix_lst.append(var_orth_matrix)


#plot two lines
plt.figure(figsize=(10, 6))
plt.plot(lambda_val_lst, var_proj_matrix_lst, marker='o' ,linewidth=2, markersize=6, alpha = 0.5, label='Variance from projection subspace')
plt.plot(lambda_val_lst, P_C_lst, marker='^', linewidth=2, linestyle = '--', markersize=6, label='Variance from projection subspace (closed form)')
plt.plot(lambda_val_lst, var_orth_matrix_lst, marker='s', linewidth=2,alpha =0.5, markersize=6, label='Variance from orthogonal subspace')
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



if __name__ == "__main__":
    # set random seeds for reproductibility
    random_state = 42
    set_random_seeds(random_state)
