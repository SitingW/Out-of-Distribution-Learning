import numpy as np

"""
functional of initparameter:
- define the distribution that initial parapmeter theta_0 is sampled from 
- what is the purpose of lambda? Used to be define the size of covariance (w.r.t. I_d). 
- if I need to sample from a multivariate normal distribution, the default covariance is identity matrix
- Otherwise, input a given covariance regardless the size. Save the lambda for other purposes.
- Question: are we going to create multiple samples from the same distribution?
- Yes, we are going to consider the multiple samples. Getting the hyperapmeter J into this class.
Create self.rng to keep all the random seed independent.
"""
class InitParameter:
    def __init__(self, dim,n_samples, mean = None,cov = None, random_state = None):
        if random_state is not None:
            #TODO: check whether the random state is not int
            self.rng = np.random.Generator(np.random.PCG64(random_state))

        if cov is None:
            cov = np.eye(dim)
        if mean is None:
            mean = np.zeros(dim)
        self.mean = mean
        self.dim = dim
        self.cov = cov
        self.n_samples = n_samples
           
    def initialization(self):

        print("Mean has NaN:", np.isnan(self.mean).any())
        print("Mean has inf:", np.isinf(self.mean).any())
        print("Cov has NaN:", np.isnan(self.cov).any())
        print("Cov has inf:", np.isinf(self.cov).any())
        return self.rng.multivariate_normal(self.mean, self.cov, size=self.n_samples).T
        