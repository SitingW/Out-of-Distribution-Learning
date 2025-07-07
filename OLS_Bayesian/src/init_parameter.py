import numpy as np

"""
functional of initparameter:
- define the distribution that initial parapmeter theta_0 is sampled from 
- what is the purpose of lambda? Used to be define the size of covariance (w.r.t. I_d). 
- if I need to sample from a multivariate normal distribution, the default covariance is identity matrix
- Otherwise, input a given covariance regardless the size. Save the lambda for other purposes.
- Question: are we going to create multiple samples from the same distribution?
- Yes, we are going to consider the multiple samples. Getting the hyperapmeter J into this class.
"""
class InitParameter:
    def __init__(self, dim,n_sample, mean = None,cov = None):
        if cov is None:
            cov = np.eye(dim)
        if mean is None:
            mean = np.zeros(dim)
        self.mean = mean
        self.dim = dim
        self.cov = cov
        self.n_sample = n_sample
           
    def initialization(self):
        mean = np.zeros(self.dim)
        cov =  self.cov
        return np.random.multivariate_normal (mean, cov, size=self.n_sample).T
        