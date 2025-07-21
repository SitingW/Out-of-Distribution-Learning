import numpy as np


class GradientDescent:
    def __init__(self,input,X_matrix, y_vector,theta_0, max_iterations, alpha, learning_rate = 0.01, lambda_val=0, reduction = 'sum'):
        self.lambda_val = lambda_val
        self.max_iterations = max_iterations
        self.theta_0 = theta_0
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.theta_history = [theta_0]
        self.f_bar_lst = [input @ theta_0]
        self.input = input
        self.X_train = X_matrix
        self.y_train = y_vector
        self.n_samples = self.X_train.shape[0]
        self.n_features = self.X_train.shape[1]
        '''
        !!!!!!!!!WARNING!!!!!!!!!!!
        The way I write the reduction is not eliminating the possible input. With high probability there will be an error in the future.
        '''
        self.reduction = reduction
        

    def predict(self, x, theta):
        return x @ theta
    
    
    #change this part into computing bunch gradients instead of one
    
    def gradient_ridge(self, x, y):
        theta_prev = self.theta_history[-1]
        prediction_error = self.predict(x, theta_prev) - y
        gradient = x.T @ prediction_error
        if self.reduction == 'mean':
            #print("n_samples", self.n_samples)
            gradient = (1 / self.n_samples) * gradient
        #what is the size of gradient?
        #print("Gradient shape: ", gradient.shape)
        ridge_term = self.lambda_val * (theta_prev - self.theta_0)
        total_gradient = gradient + ridge_term
        theta_new = theta_prev - self.learning_rate * total_gradient
        self.theta_history.append(theta_new)
        return theta_new
    
    def iterative_avg(self):
        f_bar = self.alpha * self.predict(self.input,self.theta_history[-1] ) + ( 1- self.alpha) * self.f_bar_lst[-1]
        self.f_bar_lst.append(f_bar)
        return f_bar

    def gradient_output(self):
        for i in range (self. max_iterations):
            self.gradient_ridge(self.X_train, self.y_train)
            self.iterative_avg()
        return self.f_bar_lst[-1]
    
    def solve_theta(self):
        for i in range(self.max_iterations):
            self.gradient_ridge(self.X_train, self.y_train)
        return self.theta_history[-1]