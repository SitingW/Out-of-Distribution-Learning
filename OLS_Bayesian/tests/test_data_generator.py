import pytest
import numpy as np


from data_generator import DataGenerator

def test_data_shape():
    # Initialize DataGenerator
    data_gen = DataGenerator(random_state=42)
    
    # Generate data
    n_samples = 100
    n_features = 10
    X, y, true_theta = data_gen.linear_regression_data(n_samples, n_features)
    
    # Check shapes
    assert X.shape == (n_samples, n_features), f"Expected X shape {(n_samples, n_features)}, got {X.shape}"
    assert y.shape == (n_samples,), f"Expected y shape {(n_samples,)}, got {y.shape}"
    assert true_theta.shape == (n_features,), f"Expected true_theta shape {(n_features,)}, got {true_theta.shape}"

def test_data_content():
    # Initialize DataGenerator
    data_gen = DataGenerator(random_state=42)
    
    # Generate data
    n_samples = 100
    n_features = 10
    X, y, true_theta = data_gen.linear_regression_data(n_samples, n_features)
    
    # Check if true_theta is not all zeros
    assert np.any(true_theta != 0), "Expected true_theta to have non-zero values"
    
    # Check if y is a linear combination of X and true_theta
    y_computed = X @ true_theta
    assert np.allclose(y, y_computed), "Expected y to be a linear combination of X and true_theta"