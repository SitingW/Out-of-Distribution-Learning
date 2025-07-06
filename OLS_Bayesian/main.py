from dependencies import np, plt, os

from data_generator import DataGenerator
from gradient_descent import GradientDescent
from init_parameter import InitParameter

#set random seeds
np.random.seed(42)

#define the input output space
d = 100
n = 50
o = 1 #output dimension


def inference(initial_ite,input_x, X, Y, lambda_val, max_iterations= n, alpha= 0.5, eta=0.01 ):
        f_out_lst = []
        for j in range(initial_ite):
            theta_0 = init.initialization()
            #print(theta_0)
            #input,X_matrix, y_vector, lambda_val,theta_0, max_iterations, alpha, eta
            gd = GradientDescent( input_x , X, Y,theta_0, max_iterations, alpha, eta,  lambda_val,)
            f_out_lst.append(gd.gradient_output())
            #print(gd.gradient_output())
        variance = (np.var(f_out_lst, ddof=1))
        return variance

if __name__ == "__main__":
    ##generate the parameters W*
    theta_star = np.ones(d) 
    #Generate data
    linear_gen = DataGenerator(random_state = 42)
    X, Y, theta_star = linear_gen.linear_regression_data(n_samples=n, n_features= d) 

    lambda_val = 1
  
    init = InitParameter( dim = d, lambda_val = lambda_val) 
    #input_x = np.eye(d)[30] #random generation

    #seperate the subspaces
    #I_d - X(X^TX)^{-1}X^T
    #use scipy svd get two subspaces
    def projection_matrix_qr(X):
        Q, R = np.linalg.qr(X)
        return Q @ Q.T
    P = projection_matrix_qr (X.T)
    #print("Projection matrix P shape: ", P.shape)
    U = np.eye(d) - P
    ones = np.ones(d) # Create a vector of ones with the same dimension as d
    P_X = P @ ones
    U_X = U @ ones

    initial_ite_lst = [5, 10 , 30 , 50 , 75, 100 , 200, 500] #may tune this hyperparameter
    var_P_lst = []
    var_U_lst = []
    for initial_ite in initial_ite_lst:

        var_P = inference(initial_ite,P_X, X, Y, lambda_val, max_iterations= n, alpha= 0.5, eta=0.01)
        var_P_lst.append(var_P)
        var_U = inference(initial_ite,U_X, X, Y, lambda_val, max_iterations= n, alpha= 0.5, eta=0.01)
        var_U_lst.append(var_U)

#plot two lines
plt.figure(figsize=(10, 6))
plt.plot(initial_ite_lst, var_P_lst, marker='o', linewidth=2, markersize=6, label='Variance from projection subspace')
plt.plot(initial_ite_lst, var_U_lst, marker='s', linewidth=2, markersize=6, label='Variance from orthogonal subspace')

# Customize the plot
plt.xlabel('# of initializations')
plt.ylabel('Variance')
plt.title('Variance vs Number of initializations (d=100, n=50, lambda=1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f'plots/variance_ridge_plot.png', dpi=300, bbox_inches='tight')
# Display the plot