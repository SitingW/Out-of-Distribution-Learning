import pytest
import numpy as np

from init_parameter import InitParameter

"""
purpose of this test:
- test the initialization of the InitParameter class
- test the dimension of distribution 
- test if we have the mean and covariance matrix as input
- test if the output is a vector of the correct dimension
- test if the output is a sample from the multivariate normal distribution with the given mean and covariance
- test if the output is a sample from the multivariate normal distribution with the default mean and covariance
- test if the output is a sample from the multivariate normal distribution with a given covariance
- test if the output is a sample from the multivariate normal distribution with a given mean
- test if the output is a sample from the multivariate normal distribution with a given mean and covariance

"""
#test the initialization of the InitParameter class
def test_init_parameter_initialization():
    dim = 5
    n_sample = 10
    init_param = InitParameter(dim, n_sample)
    assert init_param.n_sample == n_sample
    assert init_param.dim == dim
    assert np.array_equal(init_param.cov, np.eye(dim))
    assert np.array_equal(init_param.mean, np.zeros(dim))

#test the initialization method
def test_init_parameter_initialization_method():
    dim = 5
    nsample = 10
    init_param = InitParameter(dim, nsample)
    sample = init_param.initialization()
    assert sample.shape == (dim, nsample), "Sample should have shape (dim, n_sample)"
    assert isinstance(sample, np.ndarray), "Sample should be a numpy array"
    assert np.all(np.isfinite(sample)), "Sample should contain finite values"   

#test if we have the mean and covariance matrix as input
def test_init_parameter_with_mean_and_cov():
    dim = 5
    n_sample = 10
    #mean and covariance matrix as input
    mean = np.ones(dim)
    cov = np.eye(dim) * 2
    init_param = InitParameter(dim,n_sample, mean, cov)
    #as we have random sampling, we cannot assert the exact values, and if we want to test the mean and covariance, we need to sample multiple times
    #one sample is only one vector. Are we going to sample multiple times?
    #Do we want a for loop?

    sample = init_param.initialization()
    assert sample.shape == (dim,n_sample), "Sample should have shape (dim, n_sample)"
    assert isinstance(sample, np.ndarray)
    assert np.all(np.isfinite(sample)), "Sample should contain finite values"
    #as we have random sampling, we cannot assert the exact values, and if we want to test the mean and covariance, we need to sample multiple times
    #otherwise, even if we sample from the same distribution, we cannot guarantee the mean and covariance will be close to the input mean and covariance
    #we can only test the shape and type of the sample

def test_init_parameter_with_default_mean_and_cov():
    dim = 5
    sample_size = 10
    #default mean and covariance matrix
    init_param = InitParameter(dim, sample_size)
    sample = init_param.initialization()
    assert sample.shape == (dim, sample_size), "Sample should have shape (dim, n_sample)"
    assert isinstance(sample, np.ndarray), "Sample should be a numpy array"
    assert np.all(np.isfinite(sample)), "Sample should contain finite values"
    #test if the sample is close to the default mean and covariance
    #np.mean(sample) is computing a sample's every entry's mean, which will only be close to 0 if we have multiple samples
    assert np.allclose(np.mean(sample, axis=1), np.zeros(dim), atol=1e-1), "Sample mean should be close to 0 (this one can fail with small sample size)"
    assert np.allclose(np.cov(sample), np.eye(dim), atol=1e-5), "Sample covariance should be close to identity matrix (this one can fail with small sample size)"  


def test_init_parameter_with_given_mean_and_default_cov():
    dim = 5
    sample_size = 10
    #given mean and default covariance matrix
    mean = np.ones(dim)
    init_param = InitParameter(dim, sample_size, mean)
    sample = init_param.initialization()
    assert sample.shape == (dim, sample_size), "Sample should have shape (dim, n_sample)"
    assert isinstance(sample, np.ndarray), "Sample should be a numpy array"
    assert np.all(np.isfinite(sample)), "Sample should contain finite values"
    #test if the sample is close to the given mean and default covariance
    assert np.allclose(np.mean(sample, axis=1), mean, atol=1e-1), "Sample mean should be close to given mean (this one can fail with small sample size)"
    assert np.allclose(np.cov(sample), np.eye(dim), atol=1e-5), "Sample covariance should be close to identity matrix (this one can fail with small sample size)"  

def test_init_parameter_with_default_mean_and_given_cov():
    dim = 5
    sample_size = 10
    #default mean and given covariance matrix
    cov = np.eye(dim) * 2
    init_param = InitParameter(dim, sample_size, cov=cov)
    sample = init_param.initialization()
    assert sample.shape == (dim, sample_size), "Sample should have shape (dim, n_sample)"
    assert isinstance(sample, np.ndarray), "Sample should be a numpy array"
    assert np.all(np.isfinite(sample)), "Sample should contain finite values"
    #test if the sample is close to the default mean and given covariance
    assert np.allclose(np.mean(sample, axis=1), np.zeros(dim), atol=1e-1), "Sample mean should be close to 0 (this one can fail with small sample size)"
    assert np.allclose(np.cov(sample), cov, atol=1e-5), "Sample covariance should be close to given covariance (this one can fail with small sample size)"

def test_init_parameter_with_given_mean_and_cov():
    dim = 5
    sample_size = 10
    #given mean and covariance matrix
    mean = np.ones(dim)
    cov = np.eye(dim) * 2
    init_param = InitParameter(dim, sample_size, mean, cov)
    sample = init_param.initialization()
    assert sample.shape == (dim, sample_size), "Sample should have shape (dim, n_sample)"
    assert isinstance(sample, np.ndarray), "Sample should be a numpy array"
    assert np.all(np.isfinite(sample)), "Sample should contain finite values"
    #test if the sample is close to the given mean and covariance
    assert np.allclose(np.mean(sample, axis=1), mean, atol=1e-1), "Sample mean should be close to given mean (this one can fail with small sample size)"
    assert np.allclose(np.cov(sample), cov, atol=1e-5), "Sample covariance should be close to given covariance (this one can fail with small sample size)"
