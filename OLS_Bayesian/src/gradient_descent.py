import numpy as np


class GradientDescent:
    def __init__(self,input,X_matrix, y_vector,theta_0, max_iterations, alpha, eta, lambda_val=0):
        self.lambda_val = lambda_val
        self.max_iterations = max_iterations
        self.theta_0 = theta_0
        self.alpha = alpha
        self.eta =eta
        self.theta_history = [theta_0]
        self.f_bar_lst = [input @ theta_0]
        self.input = input
        self.X_train = X_matrix
        self.y_train = y_vector
        

    def predict(self, x, theta):
        return x @ theta
    
    
    #change this part into computing bunch gradients instead of one
   
    def gradient_ridge(self, x, y):
        theta_new = self.theta_history[-1] - self.eta * (np.mean(x.T @ (self.predict(x ,self.theta_history[-1]) - y)) + self.lambda_val * (self.theta_history[-1]-self.theta_0))
        self.theta_history.append(theta_new)
        return theta_new
    
    def iterative_avg(self):
        f_bar = self.alpha * self.predict(self.input,self.theta_history[-1] ) + ( 1- self.alpha) * self.f_bar_lst[-1]
        self.f_bar_lst.append(f_bar)

    def gradient_output(self):
        for i in range (self. max_iterations):
            self.gradient_ridge(self.X_train, self.y_train)
            self.iterative_avg()
        return self.f_bar_lst[-1]
    