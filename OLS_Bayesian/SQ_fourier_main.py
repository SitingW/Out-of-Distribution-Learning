#From the SRC files, create a dataset with one dimensional input and one dimensional output. Make sure the data is dense enough to be used for linear regression.
# The dataset should be generated using a linear function with some added noise.
import numpy as np
import matplotlib.pyplot as plt
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
    n = 500
    o = 1 #output dimension
    # Generate data
    datagen = DataGenerator(random_state=42) 
    X, Y, theta_star = datagen.get_square_wave_data(n_samples=n, n_features=d)
    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)
    #generate initial theta_0_array
    initial_ite = 50
    init_param = InitParameter(dim=d, n_sample=initial_ite)
    theta_0_array = init_param.initialization()
    #closed form solution
    config = {
    'X': X,           # np.ndarray of shape (n_samples, n_features)
    'y': Y,         # np.ndarray of shape (n_samples,)
    'lambda_val':0,   # float (regularization parameter)
    'theta_0_array': theta_0_array   # np.ndarray of shape (n_features,)
}
    closed_form_solver = ClosedFormSolver(config)
    theta_closed_form_array = closed_form_solver.compute_theta_lst()
    y_hat_closed = X @ theta_closed_form_array
    print("Closed form solution shape: ", y_hat_closed.shape)

    #gradient descent solution
    max_iterations = 500
    alpha = 0.5
    learning_rate = 0.001 
    lambda_val = 0  # Regularization parameter for gradient descent
    y_hat_gd_lst = np.zeros((n, initial_ite))
    for j in range(initial_ite):
        theta_0 = theta_0_array[:, j]  # Select the j-th initial theta vector
        #print("Theta 0 value", theta_0_array)
        #print("Theta 0 shape: ", theta_0.shape)
        #input,X_matrix, y_vector, lambda_val,theta_0_array, max_iterations, alpha, eta
        #create a placeholder like input_x, won't be used in the gradient descent, need to resolve this problem in the graident descent int he future
        input_x = np.zeros((1, d))  # Placeholder input, not used in
        gd = GradientDescent( input_x, X, Y,theta_0, max_iterations, alpha, learning_rate,  lambda_val)
        theta_gd = gd.solve_theta()
        print("gradient descent theta shape: ", theta_gd.shape)
        # Compute the output for the given input using the closed-form solution
        y_hat_gd = X.flatten() * theta_gd.flatten()
        y_hat_gd_lst[:, j] = y_hat_gd  # Store the output in the array


     # Create timesteps (1 to 500)
    timesteps = np.arange(1, 501)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each row of the matrix as a separate line
    for i in range(y_hat_closed.shape[1]):  # 50 lines
        plt.plot(timesteps, y_hat_closed[:, i], alpha=0.3, linewidth=0.8, linestyle='--', label=f'Theta {i+1} (Closed Form)')
    
    # Plot the vector (ground truth) in a distinctive color
    plt.plot(timesteps, Y, color='red', linewidth=2, alpha=0.9, label='Ground Truth Signal')
    # Plot the gradient descent output
    for i in range(y_hat_gd_lst.shape[1]):  # 50 lines
        plt.plot(timesteps, y_hat_gd_lst[:, i], color='blue', linewidth=2, alpha=0.3, label='Gradient Descent Output')
    
    # Set labels and title (you can modify these)
    plt.xlabel('')  # Leave empty for you to fill
    plt.ylabel('')  # Leave empty for you to fill
    plt.title('')   # Leave empty for you to fill
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    # Set x-axis limits
    plt.xlim(1, 500)
    
    # Add legend placeholders (you can customize)
    plt.legend([''] * y_hat_closed.shape[1] + [''], loc='best')  # Empty labels for you to fill
    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.show()