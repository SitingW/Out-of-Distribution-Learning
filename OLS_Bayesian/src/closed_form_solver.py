import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import inv as sparse_inv, spsolve
from scipy.linalg import inv as dense_inv
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import json


#handle the case of sparse and dense matrices
def universal_solve(A, b):
    if sp.issparse(A):
        return spsolve(A, b)
    else:
        return np.linalg.solve(A, b)
    

class ClosedFormSolver:
    #config: Dict[str, Any]
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the solver with data and parameters.
        
        Parameters:
        X : np.ndarray
            Design matrix of shape (n_samples, n_features).
        y : np.ndarray
            Response vector of shape (n_samples,).
        theta_0 : np.ndarray
            Prior mean vector of shape (n_features,).
        lambda_val : float
            Regularization parameter.
        """
        try:
            self.X = config['X']
            self.y = config['y']
            self.lambda_val = config['lambda_val']
            self.theta_0_array = config['theta_0_array']
        except KeyError as e:
            raise ValueError(f"Missing configuration parameter: {e}")
        #from the prompt, change the function name into solve
    def closed_form_theta (self, theta_0):
        """
        Closed-form solution for the Bayesian OLS regression.
        """
        n, d = self.X.shape
        I = np.eye(d)
        
        A = self.X.T @ self.X + self.lambda_val * I
        b = self.X.T @ self.y + self.lambda_val * theta_0
        theta = universal_solve(A, b)
        return theta   
    
    def compute_theta_lst(self):
        """
        Compute the closed form theta_hat theta values for multiple iterations.
        
        Returns:
        np.ndarray
            Array of shape (n_features, n_samples) containing initial theta values.
        """
        n_features = self.theta_0_array.shape[0]
        n_sample = self.theta_0_array.shape[1]
        theta_sol_array = np.zeros((n_features, n_sample))
        for j in range(n_sample):
            theta_0 = self.theta_0_array[:, j]  # Select the j-th initial theta vector
            # Compute the closed-form solution for the j-th initial theta
            theta_sol = self.closed_form_theta(theta_0)
            # Directly assign to the j-th column of the array
            theta_sol_array[:, j] = theta_sol
        return theta_sol_array
    
    def closed_form_sol(self, input_x):
        """
        Compute the output for a given input using the closed-form solution.
        with multiple theta_result values, we will have the output to be an array with shape (n_samples, )
        """
        theta_result_lst =  self.compute_theta_lst()
        y_result_lst = theta_result_lst.T @ input_x
        return y_result_lst
    
    def quantile_inference(self, input_x):
        output_array = self.closed_form_sol(input_x)
        # Compute the 10th, 50th and 90th percentiles
        median = np.median(output_array)
        q10 = np.percentile(output_array, 10)
        q90 = np.percentile(output_array, 90)
        return  q10, median, q90
    
    def mean_inference(self, input_x):
        output_array = self.closed_form_sol(input_x)
        # Compute the mean of the output array  
        mean = np.mean(output_array)
        # Compute the variance of the output array
        variance = np.var(output_array, ddof=1)
        return mean, variance