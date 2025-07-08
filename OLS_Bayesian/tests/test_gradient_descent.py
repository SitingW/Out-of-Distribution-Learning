import pytest
import numpy as np
from gradient_descent import GradientDescent
"""
The test mainly testing on the following functions:
- __init__      : Test the initialization of the GradientDescent class.
- predict       : Test the predict method for correct output.
- gradient_ridge: Test the gradient_ridge method for updating theta and correctness.
- iterative_avg : Test the iterative_avg method for updating f_bar_lst.
- gradient_output: Test the gradient_output method for running iterations and returning final f_bar.
- Edge cases    : Test behavior with zero iterations, zero lambda, and zero learning rate.
- Parametrized tests: Test behavior with different learning rates, alpha values, and regularization values.
- Integration and convergence tests: Test the full workflow and convergence behavior.
- Performance/stress tests: Test with a larger dataset.
- Custom markers: Use custom markers for test organization.
"""

# Fixtures
@pytest.fixture
def test_data():
    """Create test data for gradient descent."""
    return {
        'input': np.array([[1, 2], [3, 4], [5, 6]]),
        'X_matrix': np.array([[1, 2], [3, 4], [5, 6]]),
        'y_vector': np.array([1, 2, 3]),
        'theta_0': np.array([0.1, 0.2]),
        'max_iterations': 5,
        'alpha': 0.5,
        'learning_rate': 0.01,
        'lambda_val': 0.1
    }


@pytest.fixture
def gd_instance(test_data):
    """Create a GradientDescent instance with test data."""
    return GradientDescent(
        input=test_data['input'],
        X_matrix=test_data['X_matrix'],
        y_vector=test_data['y_vector'],
        theta_0=test_data['theta_0'],
        max_iterations=test_data['max_iterations'],
        alpha=test_data['alpha'],
        learning_rate=test_data['learning_rate'],
        lambda_val=test_data['lambda_val']
    )


@pytest.fixture
def simple_data():
    """Simple test data for basic tests."""
    return {
        'x': np.array([[1, 2], [3, 4]]),
        'theta': np.array([0.5, 0.3])
    }


# Test Initialization
def test_initialization(gd_instance, test_data):
    """Test that the class initializes correctly."""
    assert gd_instance.max_iterations == test_data['max_iterations']
    assert gd_instance.alpha == test_data['alpha']
    assert gd_instance.learning_rate == test_data['learning_rate']
    assert gd_instance.lambda_val == test_data['lambda_val']
    
    # Test that theta_history is initialized correctly
    assert len(gd_instance.theta_history) == 1
    np.testing.assert_array_equal(gd_instance.theta_history[0], test_data['theta_0'])
    
    # Test that f_bar_lst is initialized correctly
    assert len(gd_instance.f_bar_lst) == 1
    expected_initial_f_bar = test_data['input'] @ test_data['theta_0']
    np.testing.assert_array_equal(gd_instance.f_bar_lst[0], expected_initial_f_bar)


def test_initialization_default_params(test_data):
    """Test initialization with default parameters."""
    gd = GradientDescent(
        input=test_data['input'],
        X_matrix=test_data['X_matrix'],
        y_vector=test_data['y_vector'],
        theta_0=test_data['theta_0'],
        max_iterations=test_data['max_iterations'],
        alpha=test_data['alpha']
    )
    
    assert gd.learning_rate == 0.01  # default value
    assert gd.lambda_val == 0  # default value


# Test predict method
def test_predict(gd_instance, simple_data):
    """Test the predict method."""
    result = gd_instance.predict(simple_data['x'], simple_data['theta'])
    expected = simple_data['x'] @ simple_data['theta']
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("x_shape,theta_shape,expected_output_shape", [
    ((1, 2), (2,), (1,)),  # Single sample
    ((3, 2), (2,), (3,)),  # Multiple samples
    ((5, 3), (3,), (5,)),  # Different dimensions
])
def test_predict_shapes(gd_instance, x_shape, theta_shape, expected_output_shape):
    """Test predict method with different input shapes."""
    x = np.random.randn(*x_shape)
    theta = np.random.randn(*theta_shape)
    
    result = gd_instance.predict(x, theta)
    assert result.shape == expected_output_shape


def test_predict_shape_mismatch(gd_instance):
    """Test predict method with mismatched shapes."""
    x_wrong = np.array([[1, 2, 3]])  # 3 features
    theta_wrong = np.array([0.1, 0.2])  # 2 features
    
    with pytest.raises((ValueError, IndexError)):
        gd_instance.predict(x_wrong, theta_wrong)


# Test gradient_ridge method
def test_gradient_ridge_updates_theta(gd_instance, test_data):
    """Test that gradient_ridge updates theta_history."""
    initial_theta_count = len(gd_instance.theta_history)
    
    new_theta = gd_instance.gradient_ridge(test_data['X_matrix'], test_data['y_vector'])
    
    assert len(gd_instance.theta_history) == initial_theta_count + 1
    np.testing.assert_array_equal(gd_instance.theta_history[-1], new_theta)


def test_gradient_ridge_computation(gd_instance, test_data):
    """Test the mathematical correctness of gradient_ridge."""
    x, y = test_data['X_matrix'], test_data['y_vector']
    current_theta = gd_instance.theta_history[-1]
    
    # Manual calculation
    predictions = x @ current_theta
    gradient = x.T @ (predictions - y) + gd_instance.lambda_val * (current_theta - gd_instance.theta_0)
    expected_theta = current_theta - gd_instance.learning_rate * gradient
    
    result_theta = gd_instance.gradient_ridge(x, y)
    
    np.testing.assert_array_almost_equal(result_theta, expected_theta, decimal=10)


# Test iterative_avg method
def test_iterative_avg_updates_f_bar(gd_instance):
    """Test that iterative_avg updates f_bar_lst."""
    initial_f_bar_count = len(gd_instance.f_bar_lst)
    
    gd_instance.iterative_avg()
    
    assert len(gd_instance.f_bar_lst) == initial_f_bar_count + 1


def test_iterative_avg_computation(gd_instance):
    """Test the mathematical correctness of iterative_avg."""
    # Get initial state
    current_prediction = gd_instance.predict(gd_instance.input, gd_instance.theta_history[-1])
    current_f_bar = gd_instance.f_bar_lst[-1]
    
    # Manual calculation
    expected_f_bar = gd_instance.alpha * current_prediction + (1 - gd_instance.alpha) * current_f_bar
    
    gd_instance.iterative_avg()
    
    np.testing.assert_array_almost_equal(gd_instance.f_bar_lst[-1], expected_f_bar, decimal=10)


# Test gradient_output method
def test_gradient_output_iterations(gd_instance):
    """Test that gradient_output runs the correct number of iterations."""
    initial_theta_count = len(gd_instance.theta_history)
    initial_f_bar_count = len(gd_instance.f_bar_lst)
    
    result = gd_instance.gradient_output()
    
    # Check that the correct number of iterations were performed
    expected_theta_count = initial_theta_count + gd_instance.max_iterations
    expected_f_bar_count = initial_f_bar_count + gd_instance.max_iterations
    
    assert len(gd_instance.theta_history) == expected_theta_count
    assert len(gd_instance.f_bar_lst) == expected_f_bar_count
    
    # Check that the result is the last f_bar value
    np.testing.assert_array_equal(result, gd_instance.f_bar_lst[-1])


def test_gradient_output_returns_final_f_bar(gd_instance):
    """Test that gradient_output returns the final f_bar value."""
    result = gd_instance.gradient_output()
    assert np.array_equal(result, gd_instance.f_bar_lst[-1])


# Edge cases
def test_zero_iterations(test_data):
    """Test behavior with zero iterations."""
    gd = GradientDescent(
        input=test_data['input'],
        X_matrix=test_data['X_matrix'],
        y_vector=test_data['y_vector'],
        theta_0=test_data['theta_0'],
        max_iterations=0,
        alpha=test_data['alpha'],
        learning_rate=test_data['learning_rate']
    )
    
    result = gd.gradient_output()
    
    # Should return the initial f_bar value
    expected = test_data['input'] @ test_data['theta_0']
    np.testing.assert_array_equal(result, expected)


def test_zero_lambda(test_data):
    """Test behavior with zero regularization."""
    gd = GradientDescent(
        input=test_data['input'],
        X_matrix=test_data['X_matrix'],
        y_vector=test_data['y_vector'],
        theta_0=test_data['theta_0'],
        max_iterations=1,
        alpha=test_data['alpha'],
        learning_rate=test_data['learning_rate'],
        lambda_val=0
    )
    
    result = gd.gradient_output()
    assert result is not None


def test_zero_learning_rate(test_data):
    """Test behavior with zero learning rate."""
    gd = GradientDescent(
        input=test_data['input'],
        X_matrix=test_data['X_matrix'],
        y_vector=test_data['y_vector'],
        theta_0=test_data['theta_0'],
        max_iterations=5,
        alpha=test_data['alpha'],
        learning_rate=0,
        lambda_val=test_data['lambda_val']
    )
    
    gd.gradient_output()
    
    # Theta should not change with zero learning rate
    for theta in gd.theta_history:
        np.testing.assert_array_equal(theta, test_data['theta_0'])


# Parametrized tests
#the parameterized tests are used to test the behavior of the gradient descent algorithm with different hyperparameters
@pytest.mark.parametrize("learning_rate,iterations", [
    (0.001, 10),
    (0.01, 5),
    (0.1, 3),
])
def test_different_learning_rates(test_data, learning_rate, iterations):
    """Test behavior with different learning rates."""
    gd = GradientDescent(
        input=test_data['input'],
        X_matrix=test_data['X_matrix'],
        y_vector=test_data['y_vector'],
        theta_0=test_data['theta_0'],
        max_iterations=iterations,
        alpha=test_data['alpha'],
        learning_rate=learning_rate,
        lambda_val=test_data['lambda_val']
    )
    
    result = gd.gradient_output()
    assert result is not None
    assert len(gd.theta_history) == iterations + 1


@pytest.mark.parametrize("alpha,expected_behavior", [
    (0.0, "conservative"),  # Only uses old information
    (0.5, "balanced"),      # Balances old and new
    (1.0, "aggressive"),    # Only uses new information
])
def test_different_alpha_values(test_data, alpha, expected_behavior):
    """Test behavior with different alpha values."""
    gd = GradientDescent(
        input=test_data['input'],
        X_matrix=test_data['X_matrix'],
        y_vector=test_data['y_vector'],
        theta_0=test_data['theta_0'],
        max_iterations=1,
        alpha=alpha,
        learning_rate=test_data['learning_rate'],
        lambda_val=test_data['lambda_val']
    )
    
    initial_f_bar = test_data['input'] @ test_data['theta_0']
    gd.gradient_output()
    
    if alpha == 0.0:
        # f_bar should remain close to initial with alpha=0
        np.testing.assert_array_almost_equal(gd.f_bar_lst[-1], initial_f_bar, decimal=5)
    elif alpha == 1.0:
        # f_bar should equal the prediction with alpha=1
        expected_f_bar = gd.predict(test_data['input'], gd.theta_history[-1])
        np.testing.assert_array_almost_equal(gd.f_bar_lst[-1], expected_f_bar, decimal=10)
    else:
        # For other values, just ensure it's different from initial
        assert not np.array_equal(gd.f_bar_lst[-1], initial_f_bar)


@pytest.mark.parametrize("lambda_val", [0, 0.01, 0.1, 1.0])
def test_different_regularization_values(test_data, lambda_val):
    """Test behavior with different regularization values."""
    gd = GradientDescent(
        input=test_data['input'],
        X_matrix=test_data['X_matrix'],
        y_vector=test_data['y_vector'],
        theta_0=test_data['theta_0'],
        max_iterations=5,
        alpha=test_data['alpha'],
        learning_rate=test_data['learning_rate'],
        lambda_val=lambda_val
    )
    
    result = gd.gradient_output()
    assert result is not None
    assert len(gd.theta_history) == 6  # initial + 5 iterations


# Integration and convergence tests
def test_convergence_behavior(test_data):
    """Test that the algorithm shows convergence behavior."""
    gd = GradientDescent(
        input=test_data['input'],
        X_matrix=test_data['X_matrix'],
        y_vector=test_data['y_vector'],
        theta_0=test_data['theta_0'],
        max_iterations=100,
        alpha=test_data['alpha'],
        learning_rate=0.01,
        lambda_val=0.01
    )
    
    gd.gradient_output()
    
    # Check that theta values are changing less over time (convergence)
    early_change = np.linalg.norm(gd.theta_history[2] - gd.theta_history[1])
    late_change = np.linalg.norm(gd.theta_history[-1] - gd.theta_history[-2])
    
    # Later changes should be smaller (convergence)
    assert late_change < early_change


def test_full_workflow(test_data):
    """Test the complete workflow from initialization to final output."""
    gd = GradientDescent(
        input=test_data['input'],
        X_matrix=test_data['X_matrix'],
        y_vector=test_data['y_vector'],
        theta_0=test_data['theta_0'],
        max_iterations=test_data['max_iterations'],
        alpha=test_data['alpha'],
        learning_rate=test_data['learning_rate'],
        lambda_val=test_data['lambda_val']
    )
    
    # Test individual components
    assert len(gd.theta_history) == 1
    assert len(gd.f_bar_lst) == 1
    
    # Test gradient step
    gd.gradient_ridge(test_data['X_matrix'], test_data['y_vector'])
    assert len(gd.theta_history) == 2
    
    # Test iterative average
    gd.iterative_avg()
    assert len(gd.f_bar_lst) == 2
    
    # Test full workflow
    result = gd.gradient_output()
    assert result is not None
    assert len(gd.theta_history) == test_data['max_iterations'] + 2  # +1 initial, +1 from manual step
    assert len(gd.f_bar_lst) == test_data['max_iterations'] + 2


# Performance/stress tests
@pytest.mark.slow
def test_large_dataset():
    """Test with a larger dataset."""
    # Create larger test data
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    theta_0 = np.random.randn(n_features)
    
    gd = GradientDescent(
        input=X,
        X_matrix=X,
        y_vector=y,
        theta_0=theta_0,
        max_iterations=10,
        alpha=0.5,
        learning_rate=0.001,
        lambda_val=0.01
    )
    
    result = gd.gradient_output()
    assert result is not None
    assert len(gd.theta_history) == 11  # 10 iterations + initial


# Custom markers for test organization
@pytest.mark.unit
def test_predict_unit(gd_instance, simple_data):
    """Unit test for predict method."""
    result = gd_instance.predict(simple_data['x'], simple_data['theta'])
    expected = simple_data['x'] @ simple_data['theta']
    np.testing.assert_array_equal(result, expected)


@pytest.mark.integration
def test_gradient_descent_integration(gd_instance):
    """Integration test for the full gradient descent process."""
    result = gd_instance.gradient_output()
    assert result is not None
    assert len(gd_instance.theta_history) == gd_instance.max_iterations + 1
    assert len(gd_instance.f_bar_lst) == gd_instance.max_iterations + 1