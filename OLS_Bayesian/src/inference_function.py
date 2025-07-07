
from typing import Callable, Any

def variance(name, initial_ite,input_x, X, Y, lambda_val, max_iterations= n, alpha= 0.5, eta=0.01 ):
    f_out_lst = []
    for j in range(initial_ite):
        theta_0 = init.initialization()
        #print(theta_0)
        #input,X_matrix, y_vector, lambda_val,theta_0, max_iterations, alpha, eta
        gd = GradientDescent( input_x , X, Y, lambda_val,theta_0, max_iterations, alpha, eta)
        f_out_lst.append(gd.gradient_output())
        #print(gd.gradient_output())
        variance = (np.var(f_out_lst, ddof=1))

    return variance




def inference_function(data, method_func: Callable[[Any], float], **kwargs):
    """
    General inference function that can use different methods

    Args:
        data: Input data for inference
        method_func: Function that defines the method to use
        **kwargs: Additional arguments for the method function

    Returns:
        Result of the inference using the specified method
    """
    result = method_func(data, **kwargs)
    return result