#From the SRC files, create a dataset with one dimensional input and one dimensional output. Make sure the data is dense enough to be used for linear regression.
# The dataset should be generated using a linear function with some added noise.
import numpy as np
import os
import sys
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_generator import DataGenerator
from closed_form_solver import ClosedFormSolver
from gradient_descent import GradientDescent
from init_parameter import InitParameter 

if __name__ == "__main__":
    #set random seeds
    np.random.seed(42)
    #define the input output space
    d = 1
    n = 50
    o = 1 #output dimension
    # Generate data
    linear_gen = DataGenerator(random_state=42)
    X, Y, theta_star = linear_gen.linear_regression_data(n_samples=n, n_features=d)
    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)
