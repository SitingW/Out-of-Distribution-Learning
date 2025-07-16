from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mse(y_true, y_pred):
    """Mean Squared Error"""
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def r2(y_true, y_pred):
    """R-squared Score"""
    return r2_score(y_true, y_pred)

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber Loss"""
    residual = np.abs(y_true - y_pred)
    condition = residual <= delta
    return np.where(condition, 0.5 * residual**2, delta * residual - 0.5 * delta**2).mean()
